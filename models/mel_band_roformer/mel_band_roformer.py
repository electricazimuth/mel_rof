from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from models.mel_band_roformer.attend import Attend

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack, reduce, repeat

from librosa import filters


# helper functions

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# attention

class FeedForward(Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, rotary_embed=rotary_embed,
                          flash=flash_attn),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


# main class

class MelBandRoformer(Module):

    @beartype
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            num_bands=60,
            dim_head=64,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            flash_attn=True,
            dim_freqs_in=1025,
            sample_rate=44100,  # needed for mel filter bank from librosa
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=1,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            match_input_audio_length=False,  # if True, pad output tensor to match length of input tensor
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs),
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
            ]))

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, return_complex=True).shape[1]

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # for some reason, it doesn't include the first freq? just force a value for now

        mel_filter_bank[0][0] = 1.

        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value

        mel_filter_bank[-1, -1] = 1.

        # binary as in paper (then estimated masks are averaged for overlapping regions)

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), 'all frequencies need to be covered by all bands for now'

        repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('freqs_per_band', freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)

        # band split and mask estimator

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth
            )

            self.mask_estimators.append(mask_estimator)

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

        self.match_input_audio_length = match_input_audio_length

    # Forward pass adjusted for torch.compile compatibility
    def forward(
            self,
            raw_audio,
            target=None,
            return_loss_breakdown=False
    ):
        """
        Dimension variables used in comments:
        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2, for real/imag)
        d - feature dimension
        """

        device = raw_audio.device

        # --- Original einops: rearrange(raw_audio, 'b t -> b 1 t') ---
        # Add channel dimension if input is 2D (mono)
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1) # Shape: (b, 1, t)

        batch, channels, raw_audio_length = raw_audio.shape

        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (
                    self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        # Note: Using einops pack/unpack here. Usually compatible with compile.
        # If issues arise, replace with reshape/view.
        # Example:
        # packed_shape = raw_audio.shape[:-1]
        # raw_audio_flat = raw_audio.reshape(-1, raw_audio_length)
        raw_audio_flat, batch_audio_channel_packed_shape = pack([raw_audio], '* t')

        stft_window = self.stft_window_fn(device=device)

        stft_repr = torch.stft(raw_audio_flat, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr) # Shape: (b*s, f, t, c=2)

        # Note: Using einops pack/unpack here. Usually compatible with compile.
        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c') # Shape: (b, s, f, t, c)

        # --- Original einops: rearrange(stft_repr, 'b s f t c -> b (f s) t c') ---
        # Merge stereo / mono channel (s) into the frequency (f) dimension
        # Target shape: (b, f*s, t, c)
        b, s, f, t, c_ = stft_repr.shape
        stft_repr = stft_repr.permute(0, 2, 1, 3, 4) # Shape: (b, f, s, t, c)
        stft_repr = stft_repr.reshape(b, f * s, t, c_) # Shape: (b, fs, t, c)

        # index out all frequencies for all frequency ranges across bands ascending in one go
        batch_arange = torch.arange(batch, device=device)[..., None]

        # Account for stereo (already merged into f dimension in stft_repr)
        x = stft_repr[:, self.freq_indices] # Indexing freq dim. Shape: (b, num_freq_indices, t, c)
                                             # Note: freq_indices might select fewer than f*s frequencies

        # --- Original einops: rearrange(x, 'b f t c -> b t (f c)') ---
        # This was the likely source of the original SymInt error
        # Fold the complex (real and imag) into the frequencies dimension and swap t, f
        # Target shape: (b, t, f*c) where f is num_freq_indices
        b, f_idx, t, c_ = x.shape
        x = x.permute(0, 2, 1, 3) # Shape: (b, t, f_idx, c)
        x = x.reshape(b, t, f_idx * c_) # Shape: (b, t, f_idx*c)

        x = self.band_split(x) # Assume band_split handles the (b, t, f_idx*c) shape

        # axial / hierarchical attention

        for time_transformer, freq_transformer in self.layers:
            # --- Original einops: rearrange(x, 'b t f d -> b f t d') ---
            # Swap time (t) and frequency band (f) dimensions
            # Input shape assumed: (b, t, f_band, d) from band_split/previous iter
            x = x.permute(0, 2, 1, 3) # Shape: (b, f_band, t, d)

            # Note: Using einops pack/unpack here.
            x_packed, ps = pack([x], '* t d') # Shape: (b*f_band, t, d)
            x_tfm = time_transformer(x_packed)
            x, = unpack(x_tfm, ps, '* t d') # Shape: (b, f_band, t, d)

            # --- Original einops: rearrange(x, 'b f t d -> b t f d') ---
            # Swap frequency band (f) and time (t) dimensions back
            x = x.permute(0, 2, 1, 3) # Shape: (b, t, f_band, d)

            # Note: Using einops pack/unpack here.
            x_packed, ps = pack([x], '* f d') # Shape: (b*t, f_band, d)
            x_tfm = freq_transformer(x_packed)
            x, = unpack(x_tfm, ps, '* f d') # Shape: (b, t, f_band, d)


        num_stems = len(self.mask_estimators)

        # Input to mask_estimators is x: (b, t, f_band, d)
        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1) # Shape: (b, n, t, f_band*c_out) - Assuming output dim matches f*c

        # --- Original einops: rearrange(masks, 'b n t (f c) -> b n f t c', c=2) ---
        # Reshape mask output back into (freq_idx, complex) and permute
        # Target shape: (b, n, f_idx, t, c=2)
        b, n, t, fc_ = masks.shape
        c_val = 2 # Complex dimension size
        f_idx = fc_ // c_val # Calculate f_idx size from mask output dim
        masks = masks.reshape(b, n, t, f_idx, c_val) # Shape: (b, n, t, f_idx, c)
        masks = masks.permute(0, 1, 3, 2, 4) # Shape: (b, n, f_idx, t, c)

        # modulate frequency representation

        # --- Original einops: rearrange(stft_repr, 'b f t c -> b 1 f t c') ---
        # Add stem dimension (n=1) to original stft_repr
        # Input stft_repr shape: (b, fs, t, c)
        stft_repr = stft_repr.unsqueeze(1) # Shape: (b, 1, fs, t, c)

        # complex number multiplication
        stft_repr_complex = torch.view_as_complex(stft_repr) # Shape: (b, 1, fs, t) complex
        masks_complex = torch.view_as_complex(masks) # Shape: (b, n, f_idx, t) complex
        masks_complex = masks_complex.type(stft_repr_complex.dtype)

        # need to average the estimated mask for the overlapped frequencies

        # --- Original einops: repeat(self.freq_indices, 'f -> b n f t', b=batch, n=num_stems, t=stft_repr.shape[-1]) ---
        # Expand freq_indices for scatter_add_
        # Target shape: (b, n, f_idx, t)
        t_dim = stft_repr_complex.shape[-1] # Get time dimension size from complex view
        # Start with freq_indices shape: (f_idx,)
        scatter_indices = self.freq_indices.view(1, 1, -1, 1) # Shape: (1, 1, f_idx, 1)
        scatter_indices = scatter_indices.expand(batch, num_stems, -1, t_dim) # Shape: (b, n, f_idx, t)

        # --- Original einops: repeat(stft_repr, 'b 1 ... -> b n ...', n=num_stems) ---
        # Expand stft_repr along the stem dimension for scatter_add_ target
        # Target shape: (b, n, fs, t) complex
        # Use .expand() for memory efficiency if grads aren't needed through the expansion itself,
        # otherwise use .repeat(). Scatter target doesn't usually need gradients itself.
        stft_repr_expanded_stems = stft_repr_complex.expand(-1, num_stems, -1, -1) # Shape: (b, n, fs, t) complex

        # Use scatter_add_ with complex numbers requires viewing as real temporarily
        stft_repr_expanded_stems_real = torch.view_as_real(stft_repr_expanded_stems) # (b, n, fs, t, c)
        masks_real = torch.view_as_real(masks_complex) # (b, n, f_idx, t, c)
        scatter_indices_real = scatter_indices.unsqueeze(-1).expand(-1, -1, -1, -1, 2) # (b, n, f_idx, t, c)

        # Dim 2 is the frequency dimension (fs) in stft_repr_expanded_stems_real
        masks_summed_real = torch.zeros_like(stft_repr_expanded_stems_real).scatter_add_(2, scatter_indices_real, masks_real)
        masks_summed = torch.view_as_complex(masks_summed_real) # Shape: (b, n, fs, t) complex


        # --- Original einops: repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels) ---
        # Calculate denominator for averaging overlaps
        # Target shape: (fs, 1) where fs = f_orig * channels
        # Input num_bands_per_freq shape: (f_orig,) - assuming it's based on original frequencies before stereo merge
        denom = self.num_bands_per_freq.repeat_interleave(channels, dim=0) # Shape: (fs,)
        denom = denom.unsqueeze(-1) # Shape: (fs, 1)
        # Add dimensions for broadcasting: (1, 1, fs, 1)
        denom = denom.view(1, 1, -1, 1)

        masks_averaged = masks_summed / denom.clamp(min=1e-8) # Shape: (b, n, fs, t) complex

        # modulate stft repr with estimated mask
        # stft_repr_complex is (b, 1, fs, t), masks_averaged is (b, n, fs, t)
        # Broadcasting happens automatically on dim 1
        stft_repr_masked = stft_repr_complex * masks_averaged # Shape: (b, n, fs, t) complex

        # istft

        # --- Original einops: rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels) ---
        # Reshape for istft: Merge b, n, s dimensions; separate f and s from fs dimension
        # Target shape: (b*n*s, f_orig, t) complex
        b, n, fs, t_ = stft_repr_masked.shape
        s = self.audio_channels
        f_orig = fs // s # Calculate original frequency dimension size
        # Reshape to explicitly separate f_orig and s
        stft_repr_masked = stft_repr_masked.view(b, n, f_orig, s, t_) # Shape: (b, n, f_orig, s, t)
        # Permute to bring b, n, s together
        stft_repr_masked = stft_repr_masked.permute(0, 1, 3, 2, 4) # Shape: (b, n, s, f_orig, t)
        # Reshape to merge leading dimensions
        stft_repr_to_istft = stft_repr_masked.reshape(b * n * s, f_orig, t_) # Shape: (b*n*s, f_orig, t)

        recon_audio = torch.istft(stft_repr_to_istft, **self.stft_kwargs, window=stft_window, return_complex=False,
                                  length=istft_length) # Shape: (b*n*s, raw_t)

        # --- Original einops: rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=num_stems) ---
        # Reshape back into batch, stems, channels, time
        # Target shape: (b, n, s, raw_t)
        raw_t = recon_audio.shape[-1]
        recon_audio = recon_audio.view(batch, num_stems, self.audio_channels, raw_t) # Shape: (b, n, s, t)


        # --- Original einops: rearrange(recon_audio, 'b 1 s t -> b s t') ---
        # Remove stem dimension if num_stems is 1
        if num_stems == 1:
            recon_audio = recon_audio.squeeze(1) # Shape: (b, s, t)

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        # Ensure target has same number of stems if multi-stem output
        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems, \
                   f"Target ndim ({target.ndim}) or stems ({target.shape[1]}) mismatch. Expected 4 dims and {self.num_stems} stems."

        # --- Original einops: rearrange(target, '... t -> ... 1 t') ---
        # Add channel dimension 's' if target is missing it compared to recon_audio
        # Handles case where target is (b, t) for mono or (b, n, t) for multi-stem mono target
        # Should result in (b, 1, t) or (b, n, 1, t)
        if target.ndim == (recon_audio.ndim - 1):
             target = target.unsqueeze(-2) # Add channel dim before time


        # Protect against lost length on istft
        min_len = min(recon_audio.shape[-1], target.shape[-1])
        recon_audio = recon_audio[..., :min_len]
        target = target[..., :min_len]

        # L1 loss in time domain
        loss = F.l1_loss(recon_audio, target)

        # Multi-resolution STFT loss
        multi_stft_resolution_loss = 0.

        # Permute recon_audio and target for STFT: (... s t) -> (...*s, t)
        # --- Original einops: rearrange(recon_audio, '... s t -> (... s) t') ---
        recon_audio_flat = recon_audio.reshape(-1, recon_audio.shape[-1]) # Shape: (B*S, T) where B includes stems if present
        # --- Original einops: rearrange(target, '... s t -> (... s) t') ---
        target_flat = target.reshape(-1, target.shape[-1]) # Shape: (B*S, T)

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(recon_audio_flat, **res_stft_kwargs)
            target_Y = torch.stft(target_flat, **res_stft_kwargs)

            # L1 loss on magnitude of STFT
            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(torch.abs(recon_Y), torch.abs(target_Y))
            # Optional: Add phase loss component too? Often just magnitude is used.
            # phase_loss = F.mse_loss(torch.angle(recon_Y), torch.angle(target_Y)) # Example

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
    
