import torch
from torch import nn, einsum
from math import pi
import warnings # Import warnings

# --- Need to import necessary components from the original module ---
# Option 1: If the file is accessible (e.g., copied locally)
# from rotary_embedding_torch import RotaryEmbedding as OriginalRotaryEmbedding, \
#                                     apply_rotary_emb, \
#                                     exists, \
#                                     default

# Option 2: If using the installed package, try importing directly
try:
    from rotary_embedding_torch.rotary_embedding_torch import (
        RotaryEmbedding as OriginalRotaryEmbedding,
        apply_rotary_emb,
        exists,
        default,
        rearrange # Keep rearrange for potential use in apply_rotary_emb or elsewhere
    )
except ImportError:
    raise ImportError("Could not import from rotary_embedding_torch. Make sure it's installed.")

# --- Compile-Safe Wrapper Class ---

class RotaryEmbeddingCompileSafeWrapper(nn.Module):
    """
    A wrapper around rotary_embedding_torch.RotaryEmbedding designed for
    compatibility with torch.compile(..., mode="reduce-overhead" or "max-autotune")
    by addressing CUDA graph tensor overwriting issues related to caching.

    It achieves this by:
    1. Cloning tensors retrieved from the internal cache.
    2. Using torch.repeat_interleave instead of einops.repeat internally.
    3. Includes a _load_from_state_dict hook for compatibility with older checkpoints.
    """
    def __init__(
        self,
        dim,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        # Add any other args the original __init__ might take
        **kwargs # Catch any future arguments
    ):
        super().__init__()

        # Instantiate the original RotaryEmbedding class internally
        self.rope_instance = OriginalRotaryEmbedding(
            dim=dim,
            custom_freqs=custom_freqs,
            freqs_for=freqs_for,
            theta=theta,
            max_freq=max_freq,
            num_freqs=num_freqs,
            learned_freq=learned_freq,
            use_xpos=use_xpos,
            xpos_scale_base=xpos_scale_base,
            interpolate_factor=interpolate_factor,
            theta_rescale_factor=theta_rescale_factor,
            seq_before_head_dim=seq_before_head_dim,
            **kwargs
        )

        # Store relevant attributes needed by the wrapper logic
        self.learned_freq = self.rope_instance.learned_freq
        self.use_xpos = self.rope_instance.use_xpos # Needed for assertions
        self.interpolate_factor = self.rope_instance.interpolate_factor
        self.default_seq_dim = self.rope_instance.default_seq_dim

        # --- No need to explicitly handle self.freqs here ---
        # The self.rope_instance already registers 'freqs' internally.
        # The hook below will handle mapping the loaded state dict.
        # Expose the freqs parameter if it's learnable, for optimizer registration
        #if self.learned_freq:
        #    self.freqs = self.rope_instance.freqs
        #else:
        #    # Register as buffer if not learned, mirroring original behavior
        #    self.register_buffer('freqs', self.rope_instance.freqs, persistent=False)


    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        # Use the original instance's method or re-implement if necessary
        # return self.rope_instance.get_seq_pos(seq_len, device, dtype, offset)
        # Re-implementation is simple and avoids relying on internal method:
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    # Inside RotaryEmbeddingCompileSafeWrapper class in safe_rotary.py

    def _get_safe_freqs(self, t_lambda, cache_key):
        """
        Internal method to compute frequencies safely for compilation.
        Caching is completely disabled to avoid CUDA Graph conflicts.
        """
        # --- Caching Disabled ---
        # Ignore cache_key and should_cache logic entirely

        # --- Always Compute freqs ---
        if callable(t_lambda):
            t = t_lambda() # Execute the lambda to get the position tensor
        else:
            t = t_lambda # Assume t is already the position tensor

        # Use the freqs stored in the rope_instance
        freqs_base = self.rope_instance.freqs
        # Compute the base frequencies before repeating
        freqs_computed = einsum('..., f -> ... f', t.type(freqs_base.dtype), freqs_base)

        # Apply repeat_interleave
        freqs_final = freqs_computed.repeat_interleave(2, dim=-1)

        # Return the computed freqs (optionally clone for extra safety, though likely not needed now)
        return freqs_final # .clone() # Consider adding .clone() if errors persist even without cache

    def _get_safe_freqsPREVIOUS(self, t_lambda, cache_key):
        """
        Internal method to compute or retrieve frequencies safely for compilation.
        Replicates the logic of the original `forward` method but adds cloning.
        """
        should_cache = not self.learned_freq and exists(cache_key)

        # Access the cache of the *internal* original instance
        original_cache = self.rope_instance.cache

        if should_cache and cache_key in original_cache:
            # Clone tensor retrieved from cache. The cached version should
            # already be the final, repeated form based on original logic.
            freqs_final_from_cache = original_cache[cache_key].clone()
            return freqs_final_from_cache

        # --- Compute freqs if not cached ---
        if callable(t_lambda):
            t = t_lambda() # Execute the lambda to get the position tensor
        else:
            t = t_lambda # Assume t is already the position tensor

        # Access freqs through the instance
        freqs_base = self.rope_instance.freqs
        # Compute the base frequencies before repeating
        freqs_computed = einsum('..., f -> ... f', t.type(freqs_base.dtype), freqs_base)

        # --- CRITICAL FIX: Clone *before* repeat_interleave ---
        # This ensures the tensor entering repeat_interleave is distinct
        # within the context of the graph capture.
        # Apply repeat_interleave
        # We already tried cloning before this, didn't help. Let's try without it here.
        freqs_final_computed = freqs_computed.repeat_interleave(2, dim=-1)

        # Apply repeat_interleave using the cloned tensor
        freqs_to_cache_and_return = freqs_final_computed.clone()

        # Cache the computed *and repeated* freqs
        if should_cache:
            # Store a clone to be absolutely safe
            original_cache[cache_key] = freqs_to_cache_and_return

        return freqs_to_cache_and_return

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, freq_seq_len = None):
        """
        Rotates queries or keys using compile-safe frequency generation.
        Mirrors the signature and basic logic of the original method.
        """
        seq_dim = default(seq_dim, self.default_seq_dim)
        assert not self.use_xpos, 'use .rotate_queries_and_keys for xpos'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        # Handle target sequence length for frequency generation if provided
        target_seq_len = seq_len
        if exists(freq_seq_len):
            assert freq_seq_len >= seq_len
            target_seq_len = freq_seq_len

        # --- Use the safe frequency generation method ---
        freqs = self._get_safe_freqs(
            t_lambda=lambda: self.get_seq_pos(target_seq_len, device=device, dtype=dtype, offset=offset),
            cache_key=f'freqs:{target_seq_len}|offset:{offset}'
        )
        # --- End of safe frequency generation ---

        # Adjust freqs shape if seq_dim is -3 (seq_before_head_dim=True)
        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        # --- Apply the rotation using the original helper function ---
        # Note: We pass freqs[-seq_len:] to apply_rotary_emb implicitly inside it.
        # apply_rotary_emb handles selecting the correct slice based on t's seq_len.
        rotated_t = apply_rotary_emb(freqs, t, seq_dim = seq_dim)
        rotated_t = rotated_t.type(t.dtype)
        return rotated_t

    # --- Add other methods if needed ---
    # If your code uses rotate_queries_and_keys (for use_xpos=True) or
    # rotate_queries_with_cached_keys, you would need to override them here as well,
    # likely involving calls to _get_safe_freqs and potentially a safe scale getter.

    # Example for rotate_queries_and_keys (if needed, more complex due to scale):
    # def rotate_queries_and_keys(self, q, k, seq_dim = None):
    #     seq_dim = default(seq_dim, self.default_seq_dim)
    #     assert self.use_xpos
    #     # ... get device, dtype, seq_len ...
    #     seq = self.get_seq_pos(...)
    #     freqs = self._get_safe_freqs(lambda: seq, cache_key=...)
    #     # Need a safe scale getter as well, potentially _get_safe_scale
    #     scale = self._get_safe_scale(lambda: seq, cache_key=...)
    #     # ... apply rotations using original apply_rotary_emb ...
    #     pass # Requires implementing _get_safe_scale similar to _get_safe_freqs

    # Forward method - optional, depends if the module is called directly
    # Typically, only the rotate_* methods are called externally.
    # def forward(self, *args, **kwargs):
    #    # Decide what the default forward pass should do.
    #    # Maybe raise an error, or call rotate_queries_or_keys?
    #    raise NotImplementedError("Direct forward call not standard for RoPE, use rotate_* methods.")

    # --- State Dict Loading Hook ---
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Handles loading state dicts from checkpoints saved with the original RoPE class."""

        # Construct the key name expected in the *old* checkpoint
        old_freqs_key = prefix + 'freqs'
        # Construct the key name the *new* model structure expects
        new_freqs_key = prefix + 'rope_instance.freqs'

        # Check if the old key exists in the loaded state_dict
        if old_freqs_key in state_dict:
            # If yes, move the data to the new key name
            state_dict[new_freqs_key] = state_dict.pop(old_freqs_key)
            # print(f"Remapped RoPE key: {old_freqs_key} -> {new_freqs_key}") # Optional debug print

            # Clean up metadata tracking if in strict mode
            if strict:
                # If the old key was unexpected, it's not anymore because we handled it
                if old_freqs_key in unexpected_keys:
                    unexpected_keys.remove(old_freqs_key)
                # If the new key was missing, it's not anymore because we added it
                if new_freqs_key in missing_keys:
                    missing_keys.remove(new_freqs_key)
        elif new_freqs_key in state_dict:
            # If the new key already exists (e.g., loading a checkpoint saved *with* the wrapper), do nothing.
            pass
        else:
            # If neither key exists, let the original loader handle the missing key error.
            # However, if not learned_freq, freqs might not be in the state_dict anyway.
            if self.learned_freq:
                # Only add to missing_keys if it was expected (learned)
                 if new_freqs_key not in missing_keys: # Avoid duplicates
                    missing_keys.append(new_freqs_key)
            # If not learned_freq, it's a buffer and might not be saved, which is often okay.
            # The default loader might still add it to missing_keys if strict=True,
            # but we don't forcefully add it here unless it's a learned Parameter.


        # --- Handle potential 'scale' buffer if use_xpos=True ---
        # Although your error didn't mention it, good practice to include:
        if self.use_xpos:
            old_scale_key = prefix + 'scale'
            new_scale_key = prefix + 'rope_instance.scale'
            if old_scale_key in state_dict:
                state_dict[new_scale_key] = state_dict.pop(old_scale_key)
                if strict:
                    if old_scale_key in unexpected_keys: unexpected_keys.remove(old_scale_key)
                    if new_scale_key in missing_keys: missing_keys.remove(new_scale_key)
            # No else needed usually, buffers might not be saved/loaded strictly

        # --- Call the original hook to handle the rest of the loading ---
        # Important: We modified state_dict in place, so the original hook
        #            will now see the corrected key names.
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)