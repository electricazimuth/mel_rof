# utils.py
import time
import numpy as np
import torch
import sys
import torch.nn as nn
import json 
import subprocess as sp 
from pathlib import Path 
import typing as tp 
import tempfile 
import os 
import gc # Added for garbage collection
import inspect # Added for logging
from logging_utils import log_memory as log_memory_utils, log_tensor_memory as log_tensor_utils, trigger_gc as trigger_gc_utils

# --- Existing get_model_from_config ---
def get_model_from_config(model_type, config):
    # (Keep existing code as is)
    if model_type == 'mel_band_roformer':
        from models.mel_band_roformer import MelBandRoformer
        model = MelBandRoformer(
            **dict(config.model)
        )
    else:
        print('Unknown model: {}'.format(model_type))
        model = None
    return model

# --- Existing get_windowing_array ---
def get_windowing_array(window_size, fade_size, device):
    # (Keep existing code as is)
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)


# --- START: ADDED Audio Loading Utilities (Adapted from Demucs) ---

def _read_info(path):
    """Reads metadata using ffprobe."""
    try:
        stdout_data = sp.check_output([
            'ffprobe', "-loglevel", "panic",
            str(path), '-print_format', 'json', '-show_format', '-show_streams'
        ])
        return json.loads(stdout_data.decode('utf-8'))
    except sp.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr.decode()}") from e
    except Exception as e:
        raise RuntimeError(f"Error running ffprobe for {path}: {e}") from e


def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        return wav
    elif channels == 1:
        # Convert to mono by averaging channels.
        return wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Mono to stereo by duplicating the channel.
        return wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # More channels than requested, take the first ones.
        return wav[..., :channels, :]
    else:
        # Less channels than requested but not mono, raise error or handle differently.
        raise ValueError(f'Failed to convert audio from {src_channels} to {channels} channels.')
    # return wav # This line was incorrect, return should be inside conditions


class AudioFile:
    """
    Allows to read audio from any format supported by ffmpeg, as well as resampling or
    converting to mono on the fly.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self._info = None

    def __repr__(self):
        features = [("path", self.path)]
        try:
            if self._info or self.path.exists():
                 features.append(("samplerate", self.samplerate()))
                 features.append(("channels", self.channels()))
                 features.append(("streams", len(self)))
            else:
                 features.append(("status", "File not found or info not loaded"))
        except Exception as e:
             features.append(("error", str(e)))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"AudioFile({features_str})"

    @property
    def info(self):
        if self._info is None:
            self._info = _read_info(self.path)
        return self._info

    @property
    def duration(self):
        try:
            return float(self.info['format']['duration'])
        except Exception as e:
            print(f"Warning: Could not determine duration from ffprobe info: {e}")
            # Fallback or re-raise? For now, return 0 or raise
            # return 0.0
            raise RuntimeError(f"Could not get duration for {self.path}") from e


    @property
    def _audio_streams(self):
        if 'streams' not in self.info:
             raise RuntimeError(f"No streams found in ffprobe info for {self.path}")
        streams = [
            index for index, stream in enumerate(self.info["streams"])
            if stream.get("codec_type") == "audio" # Use .get for safety
        ]
        if not streams:
             raise RuntimeError(f"No audio streams found in {self.path}")
        return streams

    def __len__(self):
        try:
            return len(self._audio_streams)
        except Exception as e:
            print(f"Warning: Could not determine number of streams: {e}")
            return 0 # Or raise

    def channels(self, stream=0):
        """Get number of channels for a specific audio stream."""
        try:
            audio_stream_index = self._audio_streams[stream]
            return int(self.info['streams'][audio_stream_index]['channels'])
        except IndexError:
             raise RuntimeError(f"Stream index {stream} out of bounds for audio streams.") from None
        except KeyError:
             raise RuntimeError(f"Could not find channel info for stream {stream}.") from None
        except Exception as e:
             raise RuntimeError(f"Error getting channels for stream {stream}: {e}") from e

    def samplerate(self, stream=0):
        """Get sample rate for a specific audio stream."""
        try:
            audio_stream_index = self._audio_streams[stream]
            return int(self.info['streams'][audio_stream_index]['sample_rate'])
        except IndexError:
             raise RuntimeError(f"Stream index {stream} out of bounds for audio streams.") from None
        except KeyError:
             raise RuntimeError(f"Could not find sample rate info for stream {stream}.") from None
        except Exception as e:
             raise RuntimeError(f"Error getting sample rate for stream {stream}: {e}") from e

    def read(self,
             seek_time: tp.Optional[float] = None,
             duration: tp.Optional[float] = None,
             stream: int = 0, # Default to the first audio stream
             samplerate: tp.Optional[int] = None,
             channels: tp.Optional[int] = None) -> torch.Tensor:
        """
        Read audio from the file using ffmpeg.

        Args:
            seek_time (float, optional): Start time in seconds. Defaults to None (start).
            duration (float, optional): Duration to read in seconds. Defaults to None (end).
            stream (int): Audio stream index to read. Defaults to 0.
            samplerate (int, optional): Target sample rate. Resamples if different from source.
                                        Defaults to None (original sample rate).
            channels (int, optional): Target number of channels (e.g., 1 for mono, 2 for stereo).
                                      Converts if different from source. Defaults to None (original channels).

        Returns:
            torch.Tensor: Audio waveform as a tensor [channels, samples].
        """
        if stream >= len(self):
            raise ValueError(f"Stream index {stream} out of bounds. File has {len(self)} audio streams.")

        stream_channels = self.channels(stream)
        stream_samplerate = self.samplerate(stream)
        target_samplerate = samplerate or stream_samplerate
        target_channels = channels or stream_channels

        log_memory_utils("AudioFile.read start")

        # Calculate target size and query duration carefully if duration is specified
        target_size = None
        query_duration = duration
        if duration is not None:
            target_size = int(target_samplerate * duration)
            # Request slightly more from ffmpeg to ensure we get enough samples after resampling
            query_duration = float((target_size + 1) / target_samplerate)
            if query_duration == 0: # Handle edge case of very short duration
                 query_duration = 1.0 / target_samplerate # Request at least one sample's worth

        # Use a temporary file for ffmpeg output
        temp_wav_path = None
        try:
            # Create a temporary file to store ffmpeg's raw output
            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as temp_f:
                temp_wav_path = temp_f.name

            command = ['ffmpeg', '-y'] # Overwrite temporary file if it exists
            command += ['-loglevel', 'error'] # Only show errors
            if seek_time is not None:
                command += ['-ss', str(seek_time)]
            command += ['-i', str(self.path)]
            command += ['-map', f'0:{self._audio_streams[stream]}'] # Map the chosen audio stream
            if query_duration is not None:
                command += ['-t', str(query_duration)] # Limit duration
            command += ['-f', 'f32le'] # Output format: 32-bit floating point little-endian PCM
            command += ['-ar', str(target_samplerate)] # Resample if needed
            # Let ffmpeg handle channel mapping initially based on stream, convert later if needed
            command += ['-ac', str(stream_channels)] # Keep original channels for now
            command += ['-threads', '1'] # May or may not improve performance
            command += [temp_wav_path] # Output path

            log_memory_utils("Before ffmpeg call (AudioFile.read)")
            # print(f"Running ffmpeg command: {' '.join(command)}") # Debugging
            process = sp.run(command, check=True, capture_output=True) # Capture stderr
            log_memory_utils("After ffmpeg call, before reading raw (AudioFile.read)")

            # Read the raw float32 data
            wav = np.fromfile(temp_wav_path, dtype=np.float32)
            log_memory_utils("After reading raw file (AudioFile.read)")
            log_tensor_utils(wav, "wav np raw", "AudioFile.read") # Using log_tensor_utils

            wav = torch.from_numpy(wav)
            log_tensor_utils(wav, "wav_tensor_flat", "AudioFile.read")

            # Reshape according to the number of channels ffmpeg *should* have output
            # Note: ffmpeg might output fewer samples than expected for short files/durations
            num_samples = wav.numel() // stream_channels
            if wav.numel() % stream_channels != 0:
                 print(f"Warning: Raw data size {wav.numel()} not divisible by stream channels {stream_channels}. Truncating.")
                 wav = wav[:num_samples * stream_channels] # Truncate potential extra bytes

            wav = wav.view(num_samples, stream_channels).t() # Reshape to [channels, samples]
            log_tensor_utils(wav, "wav_tensor_reshaped", "AudioFile.read")

            # Perform channel conversion if necessary (using our function for consistent mono mix)
            if target_channels != stream_channels:
                log_memory_utils("Before channel conversion (AudioFile.read)")
                wav = convert_audio_channels(wav, target_channels)
                log_memory_utils("After channel conversion (AudioFile.read)")
                log_tensor_utils(wav, "wav_tensor_channels_converted", "AudioFile.read")


            # Trim to the exact target size if duration was specified
            if target_size is not None:
                log_memory_utils("Before final trim (AudioFile.read)")
                wav = wav[..., :target_size]
                log_memory_utils("After final trim (AudioFile.read)")
                log_tensor_utils(wav, "wav_tensor_final", "AudioFile.read")


            log_memory_utils("AudioFile.read end")
            return wav

        except sp.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else "No stderr"
            raise RuntimeError(f"ffmpeg failed with command:\n{' '.join(command)}\nError: {stderr}") from e
        except Exception as e:
             # Include command in error message if available
             cmd_str = f" with command:\n{' '.join(command)}" if 'command' in locals() else ""
             raise RuntimeError(f"Error reading audio file {self.path}{cmd_str}: {e}") from e
        finally:
            # Clean up the temporary file
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except OSError as e:
                    print(f"Warning: Could not delete temporary file {temp_wav_path}: {e}")


# --- END: ADDED Audio Loading Utilities ---


# --- Existing demix_track (modified to accept runtime_batch_size) ---
def demix_track(config, model, mix, device, runtime_batch_size: int, first_chunk_time=None):
    # ensure it uses runtime_batch_size
    log_memory_utils("Demix track start") # Use the imported/defined logger
    C = config.inference.chunk_size
    N = config.inference.num_overlap
    # --- USE the runtime batch size passed as an argument ---
    batch_size = runtime_batch_size # Use the argument directly
    # Ensure batch_size is at least 1 (should be handled by Cog Input, but double-check)
    if batch_size < 1:
        print(f"Warning: Received invalid batch_size ({batch_size}). Setting to 1.")
        batch_size = 1

    step = C // N
    fade_size = C // 10
    border = C - step
    total_length = mix.shape[1]
    log_tensor_utils(mix, "mix (input)", "Demix start")

    # --- Padding ---
    # Ensure mix is a tensor and on the correct device early
    padded_mix = mix # Start assuming no padding needed
    if total_length > 2 * border and border > 0:
        log_memory_utils("Before input padding")
        padded_mix = nn.functional.pad(mix, (border, border), mode='reflect')
        log_memory_utils("After input padding")
        log_tensor_utils(padded_mix, "padded_mix", "After padding")
        padded_total_length = padded_mix.shape[1]
    else:
        padded_total_length = total_length

    # --- Delete original mix tensor if padding occurred and it's different ---
    if padded_mix is not mix:
         print("Deleting original mix tensor after padding...")
         del mix
         trigger_gc_utils("After deleting original mix in demix")

    log_memory_utils("Before windowing array")
    windowing_array = get_windowing_array(C, fade_size, device)
    log_memory_utils("After windowing array")

    num_instruments = len(config.training.instruments)
    if config.training.target_instrument is not None:
        num_instruments = 1
    req_shape = (num_instruments,) + tuple(padded_mix.shape)
    log_memory_utils(f"Before creating accumulators (Shape: {req_shape})")

    result = torch.zeros(req_shape, dtype=torch.float32).to(device)
    counter = torch.zeros(req_shape, dtype=torch.float32).to(device)
    log_memory_utils("After creating accumulators (result, counter) on GPU")
    log_tensor_utils(result, "result (GPU)", "After creation")
    log_tensor_utils(counter, "counter (GPU)", "After creation")

    i = 0
    batch_data = []       # List to hold chunks for the current batch
    batch_locations = []  # List to hold (start_index, segment_length) for each chunk

    # Calculate total chunks for estimation (if needed)
    num_chunks = (padded_total_length + step - 1) // step
    first_chunk_processed_flag = (first_chunk_time is not None)

    # Check for autocast availability more robustly
    use_amp = hasattr(torch.cuda, 'amp') and torch.cuda.is_available()
    inference_context = torch.cuda.amp.autocast() if use_amp else torch.no_grad() # Use contextlib
    print(f"Starting inference loop: {num_chunks} total chunks, batch size {batch_size}")
    log_memory_utils("Start of inference loop")

    # with torch.cuda.amp.autocast() if use_amp else nullcontext(): # Alternative using nullcontext
    with inference_context:
        with torch.no_grad():
            while i < padded_total_length:
                # --- Prepare Chunk ---
                part = padded_mix[:, i : i + C]
                current_length = part.shape[-1]

                # Padding for the last chunk
                if current_length < C:
                    pad_length = C - current_length
                    # Choose padding mode based on length (consistent with original potentially)
                    mode = 'reflect' if current_length > C // 2 + 1 else 'constant'
                    part = nn.functional.pad(input=part, pad=(0, pad_length), mode=mode)

                batch_data.append(part)
                batch_locations.append((i, current_length))
                i += step

                # --- Process Batch 
                if len(batch_data) >= batch_size or i >= padded_total_length:
                    current_batch_size = len(batch_data) # Actual size of this batch
                    # log_memory_utils(f"Loop: Processing batch {len(batch_locations)//batch_size}/{num_chunks//batch_size}, Size: {current_batch_size}")

                    # --- First Chunk Timing ---
                    if not first_chunk_processed_flag:
                         chunk_start_time = time.time()

                    log_memory_utils(f"Loop: Before stack (Batch Size: {current_batch_size})")
                    batch_tensor = torch.stack(batch_data, dim=0)
                    log_memory_utils(f"Loop: After stack, Before model call")
                    log_tensor_utils(batch_tensor, "batch_tensor", f"Loop Batch {i//step//batch_size}")

                    batch_output = model(batch_tensor) # Shape: [batch, stems, chans, chunk_size]
                    log_memory_utils(f"Loop: After model call")
                    log_tensor_utils(batch_output, "batch_output", f"Loop Batch {i//step//batch_size}")


                    if not first_chunk_processed_flag:
                         chunk_time = time.time() - chunk_start_time
                         first_chunk_time = chunk_time
                         estimated_total_time = chunk_time * (num_chunks / batch_size)
                         print(f"First batch processed in {chunk_time:.2f}s. Estimated total: {estimated_total_time:.2f}s (batch size: {batch_size})")
                         first_chunk_processed_flag = True

                    # --- Process Batch Results ---
                    log_memory_utils(f"Loop: Before processing batch results")
                    for j in range(current_batch_size): # Use actual batch size
                        start_pos, seg_len = batch_locations[j]
                        # Get the output corresponding to the j-th input chunk
                        # It should have shape (num_stems, channels, time)
                        output_chunk = batch_output[j] # Get result for the j-th item in the batch

                        # --- Apply Windowing (Correctly based on overall position) ---

                        #window = windowing_array.clone() # Use clone for safety
                        # Adjust window edges based on position in the *padded* mix
                        #if start_pos == 0:
                        #    window[:fade_size] = 1.0 # First chunk, no fade-in needed
                        # Check if this chunk's *end* reaches the end of the padded mix
                        #if start_pos + C >= padded_total_length:
                        #    window[-fade_size:] = 1.0 # Last chunk, no fade-out needed
                        window = windowing_array # No clone needed if not modified in place
                        # Adjust window edges (no in-place modification)
                        current_window = window.clone() # Clone for modification safety
                        if start_pos == 0: current_window[:fade_size] = 1.0
                        if start_pos + C >= padded_total_length: current_window[-fade_size:] = 1.0

                        window_valid = current_window[:seg_len]
                        output_chunk_valid = output_chunk[..., :seg_len]
                        window_expanded_valid = window_valid.unsqueeze(0).unsqueeze(0)


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

                    log_memory_utils(f"Loop: After accumulating batch results")

                    # --- Clean up batch tensors ---
                    del batch_tensor
                    del batch_output
                    # trigger_gc_utils(f"Loop Batch Cleanup {i//step//batch_size}") # Optional: GC every batch? Maybe too slow.

                    batch_data.clear()
                    batch_locations.clear()

                if first_chunk_time is not None:
                    chunks_processed = i // step
                    batches_processed = chunks_processed / batch_size
                    batches_total = num_chunks / batch_size
                    time_remaining = first_chunk_time * (batches_total - batches_processed)
                    if time_remaining < 0: time_remaining = 0
                    sys.stdout.write(f"\rEstimated time remaining: {time_remaining:.2f} seconds ({chunks_processed}/{num_chunks} chunks @ batch size {batch_size})")
                    sys.stdout.flush()
    print() # Newline after progress bar
    log_memory_utils("End of inference loop")
    # --- Delete padded mix tensor ---
    print("Deleting padded mix tensor...")
    del padded_mix
    padded_mix = None
    trigger_gc_utils("After deleting padded mix")

    # --- Final Processing ---
    # Avoid division by zero
    #counter = torch.clamp(counter, min=1e-8)
    #estimated_sources = result / counter
    #estimated_sources = estimated_sources.cpu().numpy()
    #np.nan_to_num(estimated_sources, copy=False, nan=0.0)
    # *** IMPLEMENTATION OF OPTION 2: Move to CPU before clamp/divide ***
    log_memory_utils("Before moving accumulators to CPU")
    print("Moving accumulator tensors result/counter to CPU...")
    result_cpu = result.cpu()
    counter_cpu = counter.cpu()
    log_memory_utils("After moving accumulators to CPU")
    log_tensor_utils(result_cpu, "result_cpu", "After move")
    log_tensor_utils(counter_cpu, "counter_cpu", "After move")


    print("Deleting GPU accumulator tensors...")
    del result
    del counter
    result = None
    counter = None
    trigger_gc_utils("After deleting GPU accumulators") # Crucial step


    # Perform clamp and division on CPU
    log_memory_utils("Before clamp/divide (CPU)")
    print("Performing final clamp and division on CPU...")
    torch.clamp_(counter_cpu, min=1e-8) # In-place clamp on CPU tensor
    estimated_sources_cpu = result_cpu / counter_cpu
    log_memory_utils("After clamp/divide (CPU)")
    log_tensor_utils(estimated_sources_cpu, "estimated_sources_cpu", "After divide")


    # Clean up intermediate CPU tensors
    print("Deleting intermediate CPU tensors...")
    del result_cpu
    result_cpu = None
    del counter_cpu
    counter_cpu = None
    trigger_gc_utils("After deleting intermediate CPU tensors")


    # Convert final result to numpy
    log_memory_utils("Before converting final result to NumPy")
    estimated_sources = estimated_sources_cpu.numpy()
    log_memory_utils("After converting final result to NumPy")
    log_tensor_utils(estimated_sources, "estimated_sources (numpy)", "After conversion")
    np.nan_to_num(estimated_sources, copy=False, nan=0.0) # In-place NaN handling
    del estimated_sources_cpu # Free CPU memory
    estimated_sources_cpu = None
    trigger_gc_utils("After deleting final CPU tensor")
    
    # --- Remove Padding ---
    if total_length > 2 * border and border > 0:
        log_memory_utils("Before removing padding (NumPy)")
        print(f"Removing padding: border={border}, current shape={estimated_sources.shape}")
        estimated_sources = estimated_sources[..., border:-border]
        log_memory_utils("After removing padding (NumPy)")
        log_tensor_utils(estimated_sources, "estimated_sources (unpadded)", "After unpadding")


    # --- Prepare Output Dictionary ---
    if config.training.target_instrument is None:
        instruments = config.training.instruments
    else:
        instruments = [config.training.target_instrument]

    # Ensure the number of estimated sources matches the expected instruments
    #if estimated_sources.shape[0] != len(instruments):
    #    print(f"Warning: Number of estimated sources ({estimated_sources.shape[0]}) does not match expected instruments ({len(instruments)}). Returning sources as is.")
        # Handle this case as needed - maybe return raw array or try to match based on index?
        # For now, let's just return what we have, which might cause errors later.
        # A safer approach might be to raise an error or return fewer instruments.
        # Let's assume the first `len(instruments)` sources correspond if there's a mismatch.
    #    num_to_return = min(estimated_sources.shape[0], len(instruments))
    #    return {k: v for k, v in zip(instruments[:num_to_return], estimated_sources[:num_to_return])}, first_chunk_time
    #else:
    #     return {k: v for k, v in zip(instruments, estimated_sources)}, first_chunk_time
    # ... (instrument matching logic) ...
    output_dict = {k: v for k, v in zip(instruments, estimated_sources)}

    # Log final numpy array sizes before returning
    for k, v in output_dict.items():
        log_tensor_utils(v, f"output_dict['{k}']", "Before return")

    log_memory_utils("Demix track end")
    return output_dict, first_chunk_time # Return the dictionary containing NumPy arrays