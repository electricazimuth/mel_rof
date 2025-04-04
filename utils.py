import time
import numpy as np
import torch
import sys
import torch.nn as nn


def get_model_from_config(model_type, config):
    if model_type == 'mel_band_roformer':
        from models.mel_band_roformer import MelBandRoformer
        model = MelBandRoformer(
            **dict(config.model)
        )
    else:
        print('Unknown model: {}'.format(model_type))
        model = None

    return model


def get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)

def demix_track(config, model, mix, device, first_chunk_time=None):
    """
    Separates audio sources using overlapping chunks and batch processing.

    Args:
        config: Configuration object (should contain inference.chunk_size,
                inference.num_overlap, inference.batch_size).
        model: The PyTorch model for separation.
        mix: The input audio mixture tensor (channels, time).
        device: The device to run inference on (e.g., 'cuda:0', 'cpu').
        first_chunk_time: Optional timing information from previous runs.

    Returns:
        A tuple containing:
        - A dictionary mapping instrument names to separated waveforms (numpy arrays).
        - Updated first_chunk_time.
    """
    # --- Configuration Reading ---
    C = config.inference.chunk_size
    N = config.inference.num_overlap
    # --- NEW: Get batch_size from config, default to 1 if not present ---
    batch_size = getattr(config.inference, 'batch_size', 1)
    print(f"Using batch size: {batch_size}") # Optional: Inform user

    step = C // N
    fade_size = C // 10
    border = C - step
    total_length = mix.shape[1]

    # --- Padding ---
    # Ensure mix is a tensor and on the correct device early
    if not isinstance(mix, torch.Tensor):
         mix = torch.tensor(mix, dtype=torch.float32)
    mix = mix.to(device)

    if total_length > 2 * border and border > 0:
        # Use torch.nn.functional.pad for tensors
        mix = nn.functional.pad(mix, (border, border), mode='reflect')
        padded_total_length = mix.shape[1] # Update total length after padding
    else:
        padded_total_length = total_length

    # --- Windowing Array ---
    windowing_array = get_windowing_array(C, fade_size, device)

    # --- Determine Output Shape ---
    # Run a dummy forward pass to determine the number of output stems/instruments if needed
    # Or rely on config.training.instruments
    num_instruments = len(config.training.instruments)
    if config.training.target_instrument is not None:
        num_instruments = 1 # If only targeting one instrument

    req_shape = (num_instruments,) + tuple(mix.shape) # Use padded mix shape

    # --- Initialize Result Tensors ---
    result = torch.zeros(req_shape, dtype=torch.float32).to(device)
    counter = torch.zeros(req_shape, dtype=torch.float32).to(device)

    # --- Batch Processing Loop ---
    i = 0
    batch_data = []       # List to hold chunks for the current batch
    batch_locations = []  # List to hold (start_index, segment_length) for each chunk

    # Calculate total chunks for estimation (if needed)
    num_chunks = (padded_total_length + step - 1) // step

    # Timing setup
    first_chunk_processed_flag = (first_chunk_time is not None) # Flag to track if first chunk timing is done

    with torch.cuda.amp.autocast(dtype=torch.float16): # Use autocast if desired/configured
        with torch.no_grad():
            while i < padded_total_length:
                # --- Prepare Chunk ---
                part = mix[:, i : i + C]
                current_length = part.shape[-1]

                # Padding for the last chunk
                if current_length < C:
                    pad_length = C - current_length
                    # Choose padding mode based on length (consistent with original potentially)
                    if current_length > C // 2 + 1: # Heuristic from original
                         part = nn.functional.pad(input=part, pad=(0, pad_length), mode='reflect')
                    else:
                         part = nn.functional.pad(input=part, pad=(0, pad_length, 0, 0), mode='constant', value=0)


                # --- Add to Batch ---
                batch_data.append(part)
                batch_locations.append((i, current_length)) # Store original start and length *before* padding
                i += step

                # --- Process Batch when Full or at the End ---
                if len(batch_data) >= batch_size or i >= padded_total_length:

                    # --- First Chunk Timing ---
                    if not first_chunk_processed_flag:
                         chunk_start_time = time.time()

                    # --- Stack and Model Forward ---
                    batch_tensor = torch.stack(batch_data, dim=0)
                    #  model returns tensor directly.
                    batch_output = model(batch_tensor)
                    # --- Sanity Check (Optional but Recommended) ---
                    if batch_output.shape[0] != len(batch_data):
                        print(f"!!! WARNING: Batch output size ({batch_output.shape[0]}) doesn't match input batch size ({len(batch_data)}) !!!")
                        # Handle this potential error state if necessary

                     # --- First Chunk Timing ---
                    if not first_chunk_processed_flag:
                         chunk_time = time.time() - chunk_start_time
                         first_chunk_time = chunk_time # Store time for the first *batch*
                         estimated_total_time = chunk_time * (num_chunks / batch_size) # Rough estimate
                         print(f"First batch processed in {chunk_time:.2f}s. Estimated total: {estimated_total_time:.2f}s")
                         first_chunk_processed_flag = True

                    # --- Process Batch Results ---
                    for j in range(len(batch_data)): # Iterate through results in the batch
                        start_pos, seg_len = batch_locations[j]
                        # Get the output corresponding to the j-th input chunk
                        # It should have shape (num_stems, channels, time)
                        output_chunk = batch_output[j] # Get result for the j-th item in the batch

                        # --- Apply Windowing (Correctly based on overall position) ---
                        window = windowing_array.clone() # Use clone for safety
                        # Adjust window edges based on position in the *padded* mix
                        if start_pos == 0:
                            window[:fade_size] = 1.0 # First chunk, no fade-in needed
                        # Check if this chunk's *end* reaches the end of the padded mix
                        if start_pos + C >= padded_total_length:
                            window[-fade_size:] = 1.0 # Last chunk, no fade-out needed


                        # --- Accumulate Results ---
                        # We need to apply the window to the time dimension of the output chunk
                        # Slice the window and the output chunk down to the actual segment length
                        window_valid = window[:seg_len]             # Shape (seg_len,)
                        output_chunk_valid = output_chunk[..., :seg_len] # Shape (stems, chans, seg_len)

                        # Expand window for broadcasting
                        window_expanded_valid = window_valid.unsqueeze(0).unsqueeze(0) # Shape (1, 1, seg_len)

                        # Apply window and accumulate
                        # Ensure output_chunk covers the full window size C for broadcasting
                        # Accumulate using the valid segment length for the slice target AND the source
                        # The target slice must also use seg_len to match the source shape
                        result[..., start_pos : start_pos + seg_len] += output_chunk_valid * window_expanded_valid
                        counter[..., start_pos : start_pos + seg_len] += window_expanded_valid


                    # --- Clear Batch ---
                    batch_data.clear()
                    batch_locations.clear()

                # --- Progress Reporting (Optional but helpful) ---
                if first_chunk_time is not None:
                    chunks_processed = i // step
                    batches_processed = chunks_processed / batch_size
                    batches_total = num_chunks / batch_size
                    time_remaining = first_chunk_time * (batches_total - batches_processed)
                    if time_remaining < 0: time_remaining = 0 # Prevent negative estimates
                    sys.stdout.write(f"\rEstimated time remaining: {time_remaining:.2f} seconds ({chunks_processed}/{num_chunks} chunks)")
                    sys.stdout.flush()


    # --- Final Processing ---
    print() # Newline after progress bar
    # Avoid division by zero
    counter = torch.clamp(counter, min=1e-8)
    estimated_sources = result / counter
    estimated_sources = estimated_sources.cpu().numpy()
    np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    # --- Remove Padding ---
    if total_length > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]

    # --- Prepare Output Dictionary ---
    if config.training.target_instrument is None:
        instruments = config.training.instruments
    else:
        instruments = [config.training.target_instrument]

    # Ensure the number of estimated sources matches the expected instruments
    if estimated_sources.shape[0] != len(instruments):
        print(f"Warning: Number of estimated sources ({estimated_sources.shape[0]}) does not match expected instruments ({len(instruments)}). Returning sources as is.")
        # Handle this case as needed - maybe return raw array or try to match based on index?
        # For now, let's just return what we have, which might cause errors later.
        # A safer approach might be to raise an error or return fewer instruments.
        # Let's assume the first `len(instruments)` sources correspond if there's a mismatch.
        num_to_return = min(estimated_sources.shape[0], len(instruments))
        return {k: v for k, v in zip(instruments[:num_to_return], estimated_sources[:num_to_return])}, first_chunk_time
    else:
         return {k: v for k, v in zip(instruments, estimated_sources)}, first_chunk_time
