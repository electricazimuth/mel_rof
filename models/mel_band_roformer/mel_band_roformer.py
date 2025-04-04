import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from functools import partial # Import partial

# Keep Attend import if it's used and compile-friendly
# from models.mel_band_roformer.attend import Attend
# Using a placeholder Attend that should be compile-friendly for now
class Attend(nn.Module):
    def __init__(self, flash=True, dropout=0.):
        super().__init__()
        self.flash = flash and hasattr(F, 'scaled_dot_product_attention')
        self.dropout_p = dropout if not self.flash else 0. # Flash handles dropout

    def forward(self, q, k, v, attn_mask=None):
        if self.flash:
            # Flash Attention (PyTorch >= 2.0)
            # Input shapes: q, k, v: (b, h, n, d)
            # Note: Flash attention mask needs different format if used
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None, # Add mask logic if needed
                dropout_p=self.dropout_p,
                is_causal=False # Assuming not causal
            )
        else:
            # Manual attention calculation
            # q: (b, h, n, d), k: (b, h, m, d) -> (b, h, n, m)
            scale = q.shape[-1] ** -0.5
            sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * scale

            if exists(attn_mask):
               sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim=-1)
            attn = F.dropout(attn, p=self.dropout_p)

            # attn: (b, h, n, m), v: (b, h, m, d) -> (b, h, n, d)
            out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
            return out


from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

#from rotary_embedding_torch import RotaryEmbedding
# Import the wrapper instead
from .safe_rotary import RotaryEmbeddingCompileSafeWrapper
# Keep pack/unpack as they often work, remove others unless used elsewhere
from einops import pack, unpack #, reduce, repeat # remove reduce/repeat if not used

# Optional: May help einops pack/unpack work better with compile
from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

from librosa import filters # Keep librosa import


# helper functions

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d

# Keep pack_one/unpack_one if using pack/unpack
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
        # Note: Sometimes F.normalize has issues with compile + autocast.
        # If errors occur here, try a manual implementation:
        # variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        # x_normalized = x * torch.rsqrt(variance + 1e-5) # or other epsilon
        # return x_normalized * self.scale * self.gamma
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
        self.scale = dim_head ** -0.5 # Used in manual attention if flash=False
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads) # Gated Attention

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape # Get batch and sequence length
        h = self.heads

        x_norm = self.norm(x)

        # --- Original einops: q, k, v = rearrange(self.to_qkv(x_norm), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads) ---
        # --- Replacement using native PyTorch ---
        qkv_out = self.to_qkv(x_norm)  # Shape: (b, n, 3 * h * d)

        qkv_val = 3 # Number of tensors (Q, K, V)

        # Calculate head dimension
        # Make sure dim_inner is consistent with to_qkv output dim
        head_dim = qkv_out.shape[-1] // (qkv_val * h)

        # Optional but recommended: Add an assertion to catch shape mismatches early
        assert qkv_out.shape[-1] == qkv_val * h * head_dim, \
            f"Input dimension {qkv_out.shape[-1]} is not divisible by 3 * num_heads ({qkv_val * h})"

        # Reshape to (b, n, qkv, h, d)
        qkv_reshaped = qkv_out.view(b, n, qkv_val, h, head_dim)

        # Permute to (qkv, b, h, n, d)
        qkv_permuted = qkv_reshaped.permute(2, 0, 3, 1, 4)

        # Unpack the first dimension (qkv) into individual tensors
        q = qkv_permuted[0] # Shape: (b, h, n, d)
        k = qkv_permuted[1] # Shape: (b, h, n, d)
        v = qkv_permuted[2] # Shape: (b, h, n, d)
        # --- End of QKV Replacement ---

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v) # Shape: (b, h, n, d)

        # Gated Attention Component
        gates = self.to_gates(x_norm) # Shape: (b, n, h)

        # --- Original einops: rearrange(gates, 'b n h -> b h n 1').sigmoid() ---
        # --- Replacement using native PyTorch ---
        gates_permuted = gates.permute(0, 2, 1) # Shape: (b, h, n)
        gates_unsqueezed = gates_permuted.unsqueeze(-1) # Shape: (b, h, n, 1)
        # --- End of Gates Replacement ---

        out = out * gates_unsqueezed.sigmoid() # Apply gates

        # --- Original einops: rearrange(out, 'b h n d -> b n (h d)') ---
        # --- Replacement using native PyTorch ---
        # Permute to bring n before h: (b, n, h, d)
        out_permuted = out.permute(0, 2, 1, 3)
        # Reshape to merge h and d: (b, n, h*d)
        # Use reshape or contiguous().view()
        # Using -1 for the last dimension size is robust
        out_reshaped = out_permuted.reshape(b, n, -1) # Shape: (b, n, h*d)
        # --- End of Output Replacement ---

        return self.to_out(out_reshaped)


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
        # Use tuple for dim_inputs list in ModuleList to make it scriptable/traceable
        self.to_features = ModuleList([
            nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            ) for dim_in in dim_inputs # Use tuple here
        ])

    def forward(self, x):
        # Ensure dim_inputs is a list or tuple of integers for split
        split_sizes = list(self.dim_inputs)
        x_split = x.split(split_sizes, dim=-1)

        outs = []
        # Check if number of splits matches number of modules
        assert len(x_split) == len(self.to_features), f"Number of splits {len(x_split)} must match number of feature modules {len(self.to_features)}"

        for split_input, to_feature in zip(x_split, self.to_features):
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
        dim_hidden = dim * mlp_expansion_factor

        # Use tuple for dim_inputs list in ModuleList to make it scriptable/traceable
        self.to_freqs = ModuleList([
             nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            ) for dim_in in dim_inputs # Use tuple here
        ])


    def forward(self, x):
        # x has shape (b, t, num_bands, d)
        x_unbound = x.unbind(dim=-2) # Unbind along the band dimension

        outs = []
        assert len(x_unbound) == len(self.to_freqs), f"Number of bands {len(x_unbound)} must match number of MLPs {len(self.to_freqs)}"

        for band_features, mlp in zip(x_unbound, self.to_freqs):
            freq_out = mlp(band_features) # Should output shape (b, t, dim_in)
            outs.append(freq_out)

        # Concatenate along the last dimension to get (b, t, total_freq_dim)
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
            dim_freqs_in=1025, # Unused?
            sample_rate=44100,
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=1,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147, # // 4 of win_length 588? Or fixed?
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

        #time_rotary_embed = RotaryEmbedding(dim=dim_head)
        #freq_rotary_embed = RotaryEmbedding(dim=dim_head)
        time_rotary_embed = RotaryEmbeddingCompileSafeWrapper(dim=dim_head)
        freq_rotary_embed = RotaryEmbeddingCompileSafeWrapper(dim=dim_head)


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

        # Calculate number of frequencies from STFT parameters
        num_freqs = stft_n_fft // 2 + 1

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)
        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy).float() # Ensure float type
        # for some reason, it doesn't include the first freq? just force a value for now
        # Ensure shape matches expected frequencies
        assert mel_filter_bank.shape == (num_bands, num_freqs), \
            f"Mel filter bank shape {mel_filter_bank.shape} mismatch. Expected {(num_bands, num_freqs)}"

        # Optional: Address potential zero issues
        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value
        mel_filter_bank[0][0] = max(mel_filter_bank[0][0], 1e-8) # Avoid exactly zero if it causes issues
        mel_filter_bank[-1, -1] = max(mel_filter_bank[-1, -1], 1e-8) # Avoid exactly zero

        # binary as in paper (then estimated masks are averaged for overlapping regions)
        freqs_per_band_bool = mel_filter_bank > 0
        assert freqs_per_band_bool.any(dim=0).all(), 'All frequencies must be covered by at least one band'

        # Calculate indices based on boolean mask
        band_indices, freq_indices_in_bank = torch.where(freqs_per_band_bool)

        # Get the final frequency indices to select based on the mel bank structure
        # This seems complex, let's double-check the logic vs the original einops version
        # The goal is to get a flat tensor of frequency indices ordered by band
        # Original logic seems okay: it selects indices where freqs_per_band_bool is True
        repeated_freq_indices = torch.arange(num_freqs).unsqueeze(0).expand(num_bands, -1)
        freq_indices_to_select = repeated_freq_indices[freqs_per_band_bool]

        if stereo:
            # Original logic: repeat index, multiply by 2, add channel offset, flatten
            # Example: [10, 25] -> [[10, 10], [25, 25]] -> [[20, 20], [50, 50]] -> [[20, 21], [50, 51]] -> [20, 21, 50, 51]
            s_indices = torch.arange(self.audio_channels)
            freq_indices_to_select = freq_indices_to_select.unsqueeze(-1) * self.audio_channels + s_indices # (num_selected_freqs, s)
            freq_indices_to_select = freq_indices_to_select.view(-1) # Flatten to (num_selected_freqs * s,)

        self.register_buffer('freq_indices', freq_indices_to_select, persistent=False) # These are the indices to gather from the full STFT
        # self.register_buffer('freqs_per_band_bool', freqs_per_band_bool, persistent=False) # Might not be needed directly

        # Calculate counts for band splitting and mask averaging
        # Sum over frequencies for each band
        num_freqs_per_band = freqs_per_band_bool.sum(dim=1)
        # Sum over bands for each frequency
        num_bands_per_freq = freqs_per_band_bool.sum(dim=0)

        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False) # Shape: (num_bands,)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False) # Shape: (num_freqs,)

        # band split and mask estimator dimensions
        # Complex (2) * num freqs in that band * channels
        freqs_per_bands_with_complex_and_channels = tuple(2 * f.item() * self.audio_channels for f in num_freqs_per_band)

        # Make sure dim_inputs is a tuple of ints
        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex_and_channels
        )

        self.mask_estimators = nn.ModuleList([])
        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex_and_channels,
                depth=mask_estimator_depth
            )
            self.mask_estimators.append(mask_estimator)

        # multi-resolution stft loss setup
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft # Base n_fft? Or max of window sizes? Check paper/impl. Using base for now.
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
        f_orig - original STFT frequency dimension size (n_fft // 2 + 1)
        t - time frames in STFT
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2, for real/imag)
        d - feature dimension in transformer
        fs - frequency dimension after merging channels (f_orig * s)
        f_idx - number of selected frequency indices (potentially repeated across bands) = len(self.freq_indices)
        f_band - number of frequency bands = num_bands
        """

        device = raw_audio.device

        # Add channel dimension if input is 2D (mono)
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1) # Shape: (b, 1, t_raw)

        batch, channels, raw_audio_length = raw_audio.shape
        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert channels == self.audio_channels, f"Input audio channels ({channels}) does not match model audio channels ({self.audio_channels})"

        # === STFT ===
        # Pack batch and channels for STFT: (b, s, t_raw) -> (b*s, t_raw)
        # REplaced einops with torch alternative
        raw_audio_flat, batch_audio_channel_packed_shape = pack([raw_audio], '* t')
        stft_window = self.stft_window_fn(device=device)
        stft_repr_complex_flat = torch.stft(raw_audio_flat, **self.stft_kwargs, window=stft_window, return_complex=True) # Shape: (b*s, f_orig, t) complex
        # View as real: (b*s, f_orig, t, c=2)
        stft_repr_real_flat = torch.view_as_real(stft_repr_complex_flat)
        # Unpack batch and channels: (b*s, f_orig, t, c) -> (b, s, f_orig, t, c)
        stft_repr_real = unpack_one(stft_repr_real_flat, batch_audio_channel_packed_shape, '* f t c')

        # === Prepare for Band Processing ===
        # Merge channel (s) and frequency (f_orig) -> (fs): (b, s, f_orig, t, c) -> (b, f_orig, s, t, c) -> (b, fs, t, c)
        b, s, f_orig, t, c_ = stft_repr_real.shape
        stft_repr_fs_real = stft_repr_real.permute(0, 2, 1, 3, 4).reshape(b, f_orig * s, t, c_) # Shape: (b, fs, t, c)

        # Select frequency indices based on mel bands: (b, fs, t, c) -> (b, f_idx, t, c)
        # freq_indices has shape (f_idx,) where f_idx includes frequencies potentially repeated across bands
        x = stft_repr_fs_real[:, self.freq_indices] # Shape: (b, f_idx, t, c)

        # Fold complex dim (c) into frequency (f_idx) and swap time: (b, f_idx, t, c) -> (b, t, f_idx, c) -> (b, t, f_idx*c)
        b, f_idx_len, t, c_ = x.shape # Re-capture shapes
        x = x.permute(0, 2, 1, 3).reshape(b, t, f_idx_len * c_) # Shape: (b, t, f_idx*c)

        # === Band Splitting and Transformer ===
        # Split into bands based on dim_inputs derived from num_freqs_per_band
        # Input: (b, t, f_idx*c) -> Output: (b, t, f_band, d)
        x = self.band_split(x) # Output shape depends on BandSplit implementation

        # Axial Transformers (Time and Frequency)
        for time_transformer, freq_transformer in self.layers:
            # Time Transformer
            # Input: (b, t, f_band, d) -> (b, f_band, t, d)
            x = x.permute(0, 2, 1, 3)
            # Pack: (b, f_band, t, d) -> (b*f_band, t, d)
            x_packed_time, ps_time = pack([x], '* t d')
            # Apply Transformer: (b*f_band, t, d) -> (b*f_band, t, d)
            x_tfm_time = time_transformer(x_packed_time)
            # Unpack: (b*f_band, t, d) -> (b, f_band, t, d)
            x, = unpack(x_tfm_time, ps_time, '* t d')

            # Frequency Transformer
            # Input: (b, f_band, t, d) -> (b, t, f_band, d)
            x = x.permute(0, 2, 1, 3)
            # Pack: (b, t, f_band, d) -> (b*t, f_band, d)
            x_packed_freq, ps_freq = pack([x], '* f d')
            # Apply Transformer: (b*t, f_band, d) -> (b*t, f_band, d)
            x_tfm_freq = freq_transformer(x_packed_freq)
            # Unpack: (b*t, f_band, d) -> (b, t, f_band, d)
            x, = unpack(x_tfm_freq, ps_freq, '* f d')

        # === Mask Estimation and Application ===
        num_stems = len(self.mask_estimators)
        # Input to estimators: (b, t, f_band, d)
        # Output of stack: (b, n, t, f_band, est_out_dim) - Assuming est outputs (b,t,f_band,est_out_dim)
        # MaskEstimator currently outputs (b, t, total_freq_dim) where total_freq_dim = sum(dim_in*2 / 2) = sum(dim_in) = f_idx*c
        # Let's assume mask_estimators output the required shape for scatter_add directly: (b, n, f_idx, t, c) complex? No, likely real.
        # RETHINK MaskEstimator output shape based on `rearrange(masks, 'b n t (f c) -> b n f t c', c=2)`
        # The original rearrange implies mask_estimator output per stem is (b, t, f_idx*c)
        masks_real = torch.stack([fn(x) for fn in self.mask_estimators], dim=1) # Shape: (b, n, t, f_idx*c)

        # Reshape mask back into (f_idx, c) and permute: (b, n, t, f_idx*c) -> (b, n, t, f_idx, c) -> (b, n, f_idx, t, c)
        b, n, t, fc_ = masks_real.shape
        c_val = 2 # Complex dimension size
        f_idx_calc = fc_ // c_val # Calculate f_idx size from mask output dim
        assert f_idx_calc * c_val == fc_, "Mask output dimension not divisible by 2 (complex)"
        assert f_idx_calc == f_idx_len, f"Mask output freq dim {f_idx_calc} doesn't match gathered freq dim {f_idx_len}"

        masks_real = masks_real.view(b, n, t, f_idx_len, c_val) # Shape: (b, n, t, f_idx, c)
        masks_real = masks_real.permute(0, 1, 3, 2, 4) # Shape: (b, n, f_idx, t, c)

        # Convert mask to complex
        masks_complex = torch.view_as_complex(masks_real) # Shape: (b, n, f_idx, t) complex

        # Prepare original STFT for modulation: Add stem dim (b, fs, t, c) -> (b, 1, fs, t, c)
        stft_repr_fs_real_unsqueezed = stft_repr_fs_real.unsqueeze(1) # Shape: (b, 1, fs, t, c)
        stft_repr_fs_complex = torch.view_as_complex(stft_repr_fs_real_unsqueezed) # Shape: (b, 1, fs, t) complex

        # Match mask dtype
        masks_complex = masks_complex.type(stft_repr_fs_complex.dtype)

        # === Averaging Overlapping Masks via Scatter Add ===
        # Expand freq_indices for scatter: Target shape (b, n, f_idx, t)
        t_dim = stft_repr_fs_complex.shape[-1]
        scatter_indices = self.freq_indices.view(1, 1, -1, 1) # Shape: (1, 1, f_idx, 1)
        scatter_indices = scatter_indices.expand(b, n, -1, t_dim) # Shape: (b, n, f_idx, t)

        # Expand target tensor for scatter_add: (b, 1, fs, t) complex -> (b, n, fs, t) complex
        stft_repr_expanded_stems_complex = stft_repr_fs_complex.expand(-1, num_stems, -1, -1) # Shape: (b, n, fs, t) complex

        # Scatter add requires real view
        stft_repr_expanded_stems_real = torch.view_as_real(stft_repr_expanded_stems_complex) # (b, n, fs, t, c)
        masks_real_for_scatter = torch.view_as_real(masks_complex) # (b, n, f_idx, t, c)
        # Index needs complex dim expanded too
        scatter_indices_real = scatter_indices.unsqueeze(-1).expand(-1, -1, -1, -1, c_val) # (b, n, f_idx, t, c)

        # Perform scatter_add on frequency dimension (dim 2)
        masks_summed_real = torch.zeros_like(stft_repr_expanded_stems_real).scatter_add_(2, scatter_indices_real, masks_real_for_scatter)
        masks_summed_complex = torch.view_as_complex(masks_summed_real) # Shape: (b, n, fs, t) complex

        # Calculate denominator for averaging
        # num_bands_per_freq has shape (f_orig,)
        denom = self.num_bands_per_freq # Shape: (f_orig,)
        if self.stereo:
             # Repeat for stereo channels: (f_orig,) -> (fs,)
            denom = denom.repeat_interleave(self.audio_channels, dim=0)
        # Add dims for broadcasting: (fs,) -> (1, 1, fs, 1)
        denom = denom.view(1, 1, -1, 1)

        # Average the masks
        masks_averaged_complex = masks_summed_complex / denom.clamp(min=1e-8) # Shape: (b, n, fs, t) complex

        # Modulate original STFT: (b, 1, fs, t) * (b, n, fs, t) -> (b, n, fs, t) complex
        stft_repr_masked_complex = stft_repr_fs_complex * masks_averaged_complex

        # === Inverse STFT ===
        # Reshape for istft: (b, n, fs, t) -> (b, n, f_orig, s, t) -> (b, n, s, f_orig, t) -> (b*n*s, f_orig, t)
        b, n, fs, t_ = stft_repr_masked_complex.shape
        s = self.audio_channels
        # f_orig already defined
        stft_repr_masked_complex = stft_repr_masked_complex.view(b, n, f_orig, s, t_) # Shape: (b, n, f_orig, s, t)
        stft_repr_masked_complex = stft_repr_masked_complex.permute(0, 1, 3, 2, 4) # Shape: (b, n, s, f_orig, t)
        stft_repr_to_istft = stft_repr_masked_complex.reshape(b * n * s, f_orig, t_) # Shape: (b*n*s, f_orig, t)

        # Perform iSTFT: (b*n*s, f_orig, t) -> (b*n*s, t_raw)
        recon_audio_flat = torch.istft(stft_repr_to_istft, **self.stft_kwargs, window=stft_window, return_complex=False, length=istft_length)

        # Reshape back to stems and channels: (b*n*s, t_raw) -> (b, n, s, t_raw)
        raw_t = recon_audio_flat.shape[-1]
        recon_audio = recon_audio_flat.view(b, n, s, raw_t)

        # Squeeze stem dimension if n=1: (b, 1, s, t_raw) -> (b, s, t_raw)
        if num_stems == 1:
            recon_audio = recon_audio.squeeze(1)

        # === Loss Calculation (if target provided) ===
        if not exists(target):
            return recon_audio

        # Ensure target dimensions match output
        expected_target_ndim = recon_audio.ndim
        if target.ndim != expected_target_ndim:
             # Try adding channel dim if appropriate
            if target.ndim == expected_target_ndim - 1:
                 target = target.unsqueeze(-2) # Add channel dim before time
            else:
                raise ValueError(f"Target ndim ({target.ndim}) does not match reconstructed audio ndim ({expected_target_ndim})")

        if self.num_stems > 1:
             assert target.shape[1] == self.num_stems, \
                    f"Target stems ({target.shape[1]}) mismatch. Expected {self.num_stems} stems."

        # Trim to min length
        min_len = min(recon_audio.shape[-1], target.shape[-1])
        recon_audio = recon_audio[..., :min_len]
        target = target[..., :min_len]

        # L1 loss in time domain
        loss = F.l1_loss(recon_audio, target)

        # Multi-resolution STFT loss
        multi_stft_resolution_loss = torch.tensor(0., device=device, dtype=loss.dtype)

        # Flatten audio for multi-res STFT: (..., s, t) -> (...*s, t)
        recon_audio_mrs_flat = recon_audio.reshape(-1, recon_audio.shape[-1])
        target_mrs_flat = target.reshape(-1, target.shape[-1])

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft), # Use max(win, base_nfft) or just win? Check literature.
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs, # Hop length, normalized
            )
            try:
                recon_Y = torch.stft(recon_audio_mrs_flat, **res_stft_kwargs)
                target_Y = torch.stft(target_mrs_flat, **res_stft_kwargs)

                # L1 loss on magnitude
                multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(torch.abs(recon_Y), torch.abs(target_Y))
                # Optional: Add phase loss component too? Often just magnitude is used.
                # phase_loss = F.mse_loss(torch.angle(recon_Y), torch.angle(target_Y)) # Example
            except RuntimeError as e:
                print(f"Warning: STFT failed for window size {window_size}. Skipping this resolution. Error: {e}")
                print(f"Audio shape: {recon_audio_mrs_flat.shape}")
                continue


        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)