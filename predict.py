# predict.py
from cog import BasePredictor, Input, Path, BaseModel
import os
import sys
import tempfile
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
# import soundfile as sf # REMOVED - Using torchaudio for saving now
from ml_collections import ConfigDict
from omegaconf import OmegaConf
import torchaudio # Keep for saving and resampling
import torchaudio.transforms as T
import gc
import lameenc
from logging_utils import log_memory as log_memory_utils, log_tensor_memory as log_tensor_utils, trigger_gc as trigger_gc_utils

# Ensure utils.py is importable and contains the new AudioFile class
try:
    from utils import demix_track, get_model_from_config, AudioFile # ADDED AudioFile
except ImportError as e:
    print(f"Error importing from utils.py: {e}")
    print("Ensure 'utils.py' is in the same directory and contains 'demix_track', 'get_model_from_config', and 'AudioFile'.")
    sys.exit(1)

import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
CONFIG_PATH = "/src/configs/vocal_mbr.yaml"
MODEL_PATH = "/src/checkpoints/model.ckpt"

#CONFIG_PATH = "configs/vocal_mbr.yaml"
#MODEL_PATH = "checkpoints/model.ckpt"


MODEL_TYPE = "mel_band_roformer"
# --- Define Model Sample Rate (Crucial!) ---
# Adjust this if your model expects a different rate (e.g., 44100)
# Check your config or model documentation. Defaulting to 44.1kHz.
DEFAULT_MODEL_SAMPLE_RATE = 44100
# -----------------------------------------

# --- Existing Resampling Function ---
# (Keep the existing resample_to_16k_high_quality function as is)
def resample_to_16k_high_quality(input_wav_path: str, output_wav_path: str):
    """
    Resamples a WAV file to 16kHz mono with the highest quality settings
    inspired by librosa's 'kaiser_best'. If stereo, uses the left channel.

    Args:
        input_wav_path (str): Path to the input WAV file.
        output_wav_path (str): Path to save the resampled WAV file.
    """
    target_sample_rate = 16000
    resampling_method = "sinc_interp_kaiser"
    lowpass_filter_width = 64  # Higher value for sharper filter
    rolloff = 0.9475937167399596 # Lower rolloff reduces aliasing
    beta = 14.769656459379492  # Specific beta for kaiser_best quality
    log_memory_utils("Start resample_to_16k")
    print(f"Resampling - Loading audio file: {input_wav_path}")
    try:
        log_memory_utils("Before torchaudio.load (resample)")
        waveform, original_sample_rate = torchaudio.load(input_wav_path)
        log_memory_utils("After torchaudio.load (resample)")
        log_tensor_utils(waveform, "waveform_resample_input")
    except Exception as e:
        print(f"Resampling - Error loading file {input_wav_path}: {e}")
        raise
    print(f"Resampling - Original sample rate: {original_sample_rate} Hz")
    print(f"Resampling - Original shape: {waveform.shape}")
    if waveform.shape[0] > 1:
        print("Resampling - Input is stereo or multi-channel. Selecting left channel (channel 0).")
        waveform = waveform[0:1, :]
        print(f"Resampling - Shape after selecting left channel: {waveform.shape}")
    elif waveform.shape[0] == 1:
        print("Resampling - Input is mono.")
    else:
        print("Resampling - Error: Input waveform has 0 channels?")
        raise ValueError("Input waveform has 0 channels")

    # Ensure waveform is float32 for resampling consistency
    if waveform.dtype != torch.float32:
         print(f"Resampling - Original dtype: {waveform.dtype}. Converting to float32.")
         waveform = waveform.to(torch.float32)

    # --- Check if Resampling is Needed ---
    if original_sample_rate == target_sample_rate:
        print(f"Resampling - Input is already at {target_sample_rate} Hz. Skipping resampling, just saving (potentially mono) file.")
        resampled_waveform = waveform
    else:
        print(f"Resampling - Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz using '{resampling_method}'...")
        # --- Create Resampler Transform ---
        resampler = T.Resample(
            orig_freq=original_sample_rate,
            new_freq=target_sample_rate,
            resampling_method=resampling_method,
            lowpass_filter_width=lowpass_filter_width,
            rolloff=rolloff,
            dtype=waveform.dtype,
            beta=beta,
        )

        # --- Perform Resampling ---
        log_memory_utils("Before resampler call")
        resampled_waveform = resampler(waveform)
        print(f"Resampling - Resampled shape: {resampled_waveform.shape}")
        log_memory_utils("After resampler call")
        log_tensor_utils(resampled_waveform, "resampled_waveform")
        del waveform # Free original waveform memory
        trigger_gc_utils("After deleting original waveform in resample")

    # --- Save Output ---
    print(f"Resampling - Saving resampled audio to: {output_wav_path}")
    try:
        # Ensure output directory exists (should already exist from tempfile)
        # os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
        log_memory_utils("Before torchaudio.save (resample)")
        torchaudio.save(
            output_wav_path,
            resampled_waveform,
            target_sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        log_memory_utils("After torchaudio.save (resample)")
        print("Resampling - Complete.")
        del resampled_waveform
        trigger_gc_utils("After deleting resampled_waveform")
    except Exception as e:
        print(f"Resampling - Error saving file {output_wav_path}: {e}")
        raise
    log_memory_utils("End resample_to_16k")
# ------------------------------------

# --- Output Definition ---
class Output(BaseModel):
    vocals: Path
    instrumental: Path
    vocals_16k_mono: Path
# ----------------------------------

# --- Predictor Class ---
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        log_memory_utils("Setup start")
        print("Setting up model...")
        start_time = time.time()

        # Load configuration
        if not os.path.exists(CONFIG_PATH):
             raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
        with open(CONFIG_PATH) as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)
            # Convert to ConfigDict if necessary for get_model_from_config
            if isinstance(loaded_config, dict):
                self.config = ConfigDict(loaded_config)
            else:
                 self.config = loaded_config
        log_memory_utils("Config loaded")
        # --- Determine Model Sample Rate ---
        self.model_sr = DEFAULT_MODEL_SAMPLE_RATE # Start with default
        # Try to read from config (adjust path as needed)
        if hasattr(self.config, 'audio') and hasattr(self.config.audio, 'sample_rate'):
             self.model_sr = int(self.config.audio.sample_rate)
             print(f"Using sample rate from config: {self.model_sr} Hz")
        elif hasattr(self.config, 'inference') and hasattr(self.config.inference, 'sample_rate'):
             self.model_sr = int(self.config.inference.sample_rate)
             print(f"Using sample rate from inference config: {self.model_sr} Hz")
        else:
             print(f"Warning: Sample rate not found in config. Using default: {self.model_sr} Hz")
        # -----------------------------------

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device.")
            log_memory_utils("Device set to CUDA")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU.")

        print(f"Loading model type: {MODEL_TYPE}")
        self.model = get_model_from_config(MODEL_TYPE, self.config)
        log_memory_utils("Model structure created (on CPU)")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weights file not found at {MODEL_PATH}")
        print(f"Loading model weights from: {MODEL_PATH}")
        # Load to CPU first, then move to device
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location='cpu'), strict=False # Added strict=False potentially
        )
        log_memory_utils("Model weights loaded (on CPU)")
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        log_memory_utils("Model moved to device")
        self.model.eval()
        log_memory_utils("Model set to eval()")

        # Determine target instrument
        self.target_instrument = 'vocals' # Default
        if hasattr(self.config, 'training') and hasattr(self.config.training, 'target_instrument') and self.config.training.target_instrument is not None:
             self.target_instrument = self.config.training.target_instrument
        elif hasattr(self.config, 'inference') and hasattr(self.config.inference, 'target_instrument') and self.config.inference.target_instrument is not None:
             self.target_instrument = self.config.inference.target_instrument
        print(f"Target instrument for separation: {self.target_instrument}")

        # Ensure torch backend is benchmarked if using CUDA
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            log_memory_utils("cudnn.benchmark enabled")

        print(f"Setup complete in {time.time() - start_time:.2f} seconds.")
        log_memory_utils("Setup complete")

    def predict(
        self,
        audio: Path = Input(description="Input audio file (any format supported by ffmpeg: wav, mp3, flac, etc.)"), # Updated description
        batch_size: int = Input(
            default=1,
            description="Batch size for inference. Impacts memory usage and potentially speed. Must be >= 1.",
            ge=1
        )
    ) -> Output:
        """Run stem separation on a single audio file."""
        log_memory_utils("Predict start")
        print(f"Processing audio file: {os.path.basename(str(audio))}")
        print(f"Using inference batch size: {batch_size}")
        start_time = time.time()
        output_dir = tempfile.mkdtemp()

        # Define variables outside try for finally block cleanup
        mixture_tensor_gpu = None
        mixture_np = None
        target_stem_np = None
        instrumental_np = None
        vocals_save_tensor = None
        instrumental_save_tensor = None
        res = None

        try:
            # --- Use AudioFile for loading ---
            print(f"Loading audio using ffmpeg (target SR: {self.model_sr} Hz, target channels: 2)...")
            log_memory_utils("Before AudioFile init")
            audio_file = AudioFile(audio)
            original_channels = audio_file.channels() # Get original channels info
            original_mono = (original_channels == 1)
            print(f"Original file channels: {original_channels}")
            log_memory_utils("After AudioFile init / info")

            # Load, resample to model_sr, convert to stereo if needed
            # Returns a torch.Tensor [channels, samples] on CPU
            log_memory_utils("Before AudioFile.read")
            mixture_tensor_cpu = audio_file.read(
                samplerate=self.model_sr,
                channels=2 # Force stereo for the model
            )
            log_memory_utils("After AudioFile.read (CPU)")
            log_tensor_utils(mixture_tensor_cpu, "mixture_tensor_cpu", "After read")
            print(f"Audio loaded via ffmpeg: shape={mixture_tensor_cpu.shape}, sample_rate={self.model_sr}, dtype={mixture_tensor_cpu.dtype}")

            # Move to target device
            log_memory_utils("Before mixture_tensor_cpu move to GPU")
            mixture_tensor_gpu = mixture_tensor_cpu.to(self.device)
            log_memory_utils("After mixture_tensor move to GPU")
            log_tensor_utils(mixture_tensor_gpu, "mixture_tensor_gpu", "After move")
            # ---------------------------------

            # --- No need for manual mono conversion or tensor creation ---
            # original_mix = mix.copy() # We'll use the mixture_tensor for subtraction
            # if len(mix.shape) == 1: ... # Handled by AudioFile.read
            # mixture_tensor = torch.tensor(mix.T...) # Handled by AudioFile.read

            # Perform separation
            print("Starting demixing process...")
            log_memory_utils("Before demix_track call")
            res, _ = demix_track(
                config=self.config,
                model=self.model,
                mix=mixture_tensor_gpu, # Pass the GPU tensor
                device=self.device,
                runtime_batch_size=batch_size,
                first_chunk_time=None
            )
            log_memory_utils("After demix_track call")
            print("Demixing complete.")

            # Result from demix_track is now a numpy array [stems, channels, samples] on CPU
            if self.target_instrument not in res:
                 raise ValueError(f"Target instrument '{self.target_instrument}' not found in model output keys: {list(res.keys())}")

            target_stem_np = res[self.target_instrument] # NumPy array [C, T] on CPU
            if not isinstance(target_stem_np, np.ndarray):
                 raise TypeError(f"Expected numpy array from demix_track, got {type(target_stem_np)}")

            log_tensor_utils(target_stem_np, "target_stem_np (from demix)", "After demix")

            # --- Delete the large GPU mixture tensor - we don't need it anymore ---
            print("Deleting GPU mixture tensor...")
            del mixture_tensor_gpu
            mixture_tensor_gpu = None # Ensure reference is gone
            trigger_gc_utils("After deleting mixture_tensor_gpu")
            log_memory_utils("After deleting mixture_tensor_gpu")


            # --- Calculate instrumental using NumPy on CPU ---
            print("Calculating instrumental stem on CPU...")
            log_memory_utils("Before instrumental calculation (CPU)")
            # Use the CPU tensor we kept from loading
            mixture_np = mixture_tensor_cpu.numpy() # [C, T]
            log_tensor_utils(mixture_np, "mixture_np (for subtraction)", "Before subtraction")
            log_tensor_utils(target_stem_np, "target_stem_np (for subtraction)", "Before subtraction")


            # Ensure shapes match for subtraction
            min_len = min(mixture_np.shape[1], target_stem_np.shape[1])
            if mixture_np.shape[1] != target_stem_np.shape[1]:
                 print(f"Warning: Length mismatch for subtraction: Input {mixture_np.shape[1]}, Stem {target_stem_np.shape[1]}. Truncating to {min_len}.")

            # Perform subtraction using NumPy
            instrumental_np = mixture_np[:, :min_len] - target_stem_np[:, :min_len] # [C, T]
            log_memory_utils("After instrumental calculation (CPU)")
            log_tensor_utils(instrumental_np, "instrumental_np", "After subtraction")

            # --- Clean up large CPU arrays used for subtraction ---
            print("Deleting CPU mixture tensor used for subtraction...")
            del mixture_tensor_cpu
            mixture_tensor_cpu = None
            del mixture_np
            mixture_np = None
            # Keep target_stem_np and instrumental_np for saving
            trigger_gc_utils("After deleting CPU mixture for subtraction")


            # --- Prepare outputs for saving ---
            log_memory_utils("Before preparing outputs for saving")
            # target_stem_np is [C, T], transpose to [T, C] for potential mono slicing
            target_stem_output_np = target_stem_np[:, :min_len].T
            # instrumental_np is [C, T], transpose to [T, C]
            instrumental_output_np = instrumental_np.T

            if original_mono:
                print("Original was mono. Taking first channel of separated stem for output.")
                target_stem_output_np = target_stem_output_np[:, 0] # Now [T]

            log_tensor_utils(target_stem_output_np, "target_stem_output_np", "Ready for save")
            log_tensor_utils(instrumental_output_np, "instrumental_output_np", "Ready for save")

            # --- Save primary outputs using torchaudio ---
            print("Saving primary output stems...")
            vocals_path_str = os.path.join(output_dir, "vocals.wav")
            instrumental_path_str = os.path.join(output_dir, "instrumental.wav")

            # Convert NumPy arrays back to tensors for torchaudio.save
            # Ensure correct dtype (float32)
            log_memory_utils("Before converting save tensors")
            vocals_save_tensor = torch.from_numpy(target_stem_output_np.astype(np.float32))
            instrumental_save_tensor = torch.from_numpy(instrumental_output_np.astype(np.float32))
            log_memory_utils("After converting save tensors")
            log_tensor_utils(vocals_save_tensor, "vocals_save_tensor", "Before save")
            log_tensor_utils(instrumental_save_tensor, "instrumental_save_tensor", "Before save")


            # Handle shapes for torchaudio (needs [C, T] or [T])
            if vocals_save_tensor.ndim == 1: # Mono case
                vocals_save_tensor = vocals_save_tensor.unsqueeze(0) # Add channel dim -> [1, T]
            else:
                vocals_save_tensor = vocals_save_tensor.t() # Stereo [T, C] -> [C, T]

            if instrumental_save_tensor.ndim == 1: # Mono case (if implemented above)
                instrumental_save_tensor = instrumental_save_tensor.unsqueeze(0)
            else:
                instrumental_save_tensor = instrumental_save_tensor.t() # Stereo [T, C] -> [C, T]

            log_tensor_utils(vocals_save_tensor, "vocals_save_tensor (shaped)", "Before save")
            log_tensor_utils(instrumental_save_tensor, "instrumental_save_tensor (shaped)", "Before save")


            log_memory_utils("Before torchaudio.save (vocals)")
            torchaudio.save(vocals_path_str, vocals_save_tensor, self.model_sr, encoding="PCM_F", bits_per_sample=32)
            log_memory_utils("After torchaudio.save (vocals)")
            print(f"Vocals saved to: {vocals_path_str} (Sample Rate: {self.model_sr} Hz)")
            del vocals_save_tensor # Delete tensor after saving
            vocals_save_tensor = None
            trigger_gc_utils("After saving vocals")


            log_memory_utils("Before torchaudio.save (instrumental)")
            torchaudio.save(instrumental_path_str, instrumental_save_tensor, self.model_sr, encoding="PCM_F", bits_per_sample=32)
            log_memory_utils("After torchaudio.save (instrumental)")
            print(f"Instrumental saved to: {instrumental_path_str} (Sample Rate: {self.model_sr} Hz)")
            del instrumental_save_tensor # Delete tensor after saving
            instrumental_save_tensor = None
            trigger_gc_utils("After saving instrumental")


            # Delete the large numpy arrays now
            print("Deleting NumPy arrays used for saving...")
            del target_stem_np
            target_stem_np = None
            del instrumental_np
            instrumental_np = None
            del target_stem_output_np
            target_stem_output_np = None
            del instrumental_output_np
            instrumental_output_np = None
            trigger_gc_utils("After deleting NumPy save arrays")


            # --- Resample Vocals to 16k Mono High quality for Whisper / Gladia ---
            vocals_16k_mono_path_str = os.path.join(output_dir, "vocals_16k_mono.wav")
            print(f"\nStarting high-quality resampling of vocals to 16k mono...")
            log_memory_utils("Before resample_to_16k call")
            resample_to_16k_high_quality(vocals_path_str, vocals_16k_mono_path_str)
            log_memory_utils("After resample_to_16k call")
            print(f"Resampled vocals saved to: {vocals_16k_mono_path_str}\n")
            # --------------------------------------------

            end_time = time.time()
            print(f"Prediction finished in {end_time - start_time:.2f} seconds.")
            log_memory_utils("Predict end")
            log_memory_utils("Predict end Summary", print_summary=True, do_output=True) # Optional final summary


            # Return paths to the saved files using cog.Path
            return Output(
                vocals=Path(vocals_path_str),
                instrumental=Path(instrumental_path_str),
                vocals_16k_mono=Path(vocals_16k_mono_path_str)
            )

        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"!!!!!!!!!! Error during prediction !!!!!!!!!")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            log_memory_utils("ERROR occured", print_summary=True) # Log memory state at error
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            raise # Re-raise the exception
        finally:
             # --- Cleanup ---
             print("Running finally block cleanup...")
             log_memory_utils("Finally block start")
             del mixture_tensor_gpu
             del mixture_np
             del target_stem_np
             del instrumental_np
             del vocals_save_tensor
             del instrumental_save_tensor
             del res # Delete the dict containing numpy arrays if it wasn't cleared before
             trigger_gc_utils("Finally block cleanup")
             log_memory_utils("Finally block end")