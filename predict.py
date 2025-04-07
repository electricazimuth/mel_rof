from cog import BasePredictor, Input, Path, BaseModel
import os
import sys
import tempfile
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from ml_collections import ConfigDict
from omegaconf import OmegaConf

# --- ADDED Imports for Resampling ---
import torchaudio
import torchaudio.transforms as T
# ------------------------------------

# Ensure utils.py is importable
# If utils.py contains relative imports, adjust sys.path or refactor utils.py
# Assuming utils.py is in the same directory:
try:
    from utils import demix_track, get_model_from_config
except ImportError:
    print("Error: Ensure 'utils.py' is in the same directory and contains 'demix_track' and 'get_model_from_config'.")
    sys.exit(1)

import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
CONFIG_PATH = "/src/configs/vocal_mbr.yaml"
MODEL_PATH = "/src/checkpoints/model.ckpt"
MODEL_TYPE = "mel_band_roformer"

# --- ADDED Resampling Function ---
def resample_to_16k_high_quality(input_wav_path: str, output_wav_path: str):
    """
    Resamples a WAV file to 16kHz mono with the highest quality settings
    inspired by librosa's 'kaiser_best'. If stereo, uses the left channel.

    Args:
        input_wav_path (str): Path to the input WAV file.
        output_wav_path (str): Path to save the resampled WAV file.
    """
    target_sample_rate = 16000

    # --- High-Quality Resampling Parameters (emulating librosa's kaiser_best) ---
    # Reference: https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best
    resampling_method = "sinc_interp_kaiser"
    lowpass_filter_width = 64  # Higher value for sharper filter
    rolloff = 0.9475937167399596 # Lower rolloff reduces aliasing
    beta = 14.769656459379492  # Specific beta for kaiser_best quality

    print(f"Resampling - Loading audio file: {input_wav_path}")
    try:
        waveform, original_sample_rate = torchaudio.load(input_wav_path)
    except Exception as e:
        print(f"Resampling - Error loading file {input_wav_path}: {e}")
        raise # Re-raise to signal failure

    print(f"Resampling - Original sample rate: {original_sample_rate} Hz")
    print(f"Resampling - Original shape: {waveform.shape}") # (channels, samples)

    # --- Handle Channels ---
    # Check if stereo or multi-channel and select the left channel (index 0)
    if waveform.shape[0] > 1:
        print("Resampling - Input is stereo or multi-channel. Selecting left channel (channel 0).")
        waveform = waveform[0:1, :] # Select first channel, keep dimensions (1, samples)
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
            dtype=waveform.dtype, # Use the waveform's dtype (should be float32 now)
            beta=beta,
        )

        # --- Perform Resampling ---
        resampled_waveform = resampler(waveform)
        print(f"Resampling - Resampled shape: {resampled_waveform.shape}")

    # --- Save Output ---
    print(f"Resampling - Saving resampled audio to: {output_wav_path}")
    try:
        # Ensure output directory exists (should already exist from tempfile)
        # os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

        torchaudio.save(
            output_wav_path,
            resampled_waveform,
            target_sample_rate,
            encoding="PCM_S", # Standard WAV encoding (signed 16-bit PCM)
            bits_per_sample=16,
        )
        print("Resampling - Complete.")
    except Exception as e:
        print(f"Resampling - Error saving file {output_wav_path}: {e}")
        raise # Re-raise to signal failure
# ------------------------------------


# --- MODIFIED Output Definition ---
class Output(BaseModel):
    vocals: Path
    instrumental: Path
    vocals_16k_mono: Path # Added field for the resampled vocals
# ----------------------------------

# --- Predictor Class ---
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
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

        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU. This might be slow.")

        # Instantiate model
        print(f"Loading model type: {MODEL_TYPE}")
        self.model = get_model_from_config(MODEL_TYPE, self.config)

        # Load model weights
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weights file not found at {MODEL_PATH}")
        print(f"Loading model weights from: {MODEL_PATH}")
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        )

        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Check for target instrument (assuming 'vocals' if not specified)
        self.target_instrument = 'vocals' # Default
        if hasattr(self.config, 'training') and hasattr(self.config.training, 'target_instrument') and self.config.training.target_instrument is not None:
             self.target_instrument = self.config.training.target_instrument
        elif hasattr(self.config, 'inference') and hasattr(self.config.inference, 'target_instrument') and self.config.inference.target_instrument is not None:
             self.target_instrument = self.config.inference.target_instrument

        print(f"Target instrument for separation: {self.target_instrument}")

        # Ensure torch backend is benchmarked if using CUDA
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        print(f"Setup complete in {time.time() - start_time:.2f} seconds.")


    def predict(
        self,
        audio: Path = Input(description="Input audio file (.wav format recommended)"),

        batch_size: int = Input(
            default=1,
            description="Batch size for inference. Impacts memory usage and potentially speed. Must be >= 1.",
            ge=1 # Add validation: must be greater than or equal to 1
        )
        # ------------------------------
    ) -> Output:
        """Run stem separation on a single audio file."""
        print(f"Processing audio file: {os.path.basename(str(audio))}")
        print(f"Using inference batch size: {batch_size}") # Inform user
        start_time = time.time()
        output_dir = tempfile.mkdtemp() # Create temp dir once

        try:
            # Read audio file
            mix, sr = sf.read(str(audio))
            print(f"Audio loaded: shape={mix.shape}, sample_rate={sr}")

            # Store original mix for instrumental calculation
            original_mix = mix.copy()

            # Handle mono input -> convert to stereo for processing
            original_mono = False
            if len(mix.shape) == 1:
                print("Input is mono, duplicating channels for processing.")
                original_mono = True
                mix = np.stack([mix, mix], axis=-1)
            elif mix.shape[1] != 2:
                 print(f"Warning: Input has {mix.shape[1]} channels. Taking first two.")
                 mix = mix[:, :2]

            # Convert to torch tensor (needs to be [channels, samples])
            mixture_tensor = torch.tensor(mix.T, dtype=torch.float32).to(self.device)

            # Perform separation using the utility function
            print("Starting demixing process...")
            # --- PASS runtime_batch_size to demix_track ---
            res, _ = demix_track(
                config=self.config,
                model=self.model,
                mix=mixture_tensor,
                device=self.device,
                runtime_batch_size=batch_size, # Pass the input value here
                first_chunk_time=None
            )
            # -------------------------------------------
            print("Demixing complete.")

            # --- Process Results ---
            if self.target_instrument not in res:
                 raise ValueError(f"Target instrument '{self.target_instrument}' not found in model output keys: {list(res.keys())}")

            target_stem_np = res[self.target_instrument] # numpy array returned from demix_track

            # Check if it's actually a numpy array (should be)
            if not isinstance(target_stem_np, np.ndarray):
                 raise TypeError(f"Expected numpy array from demix_track for '{self.target_instrument}', but got {type(target_stem_np)}")

            # Assuming the numpy array is shape [channels, samples], transpose to [samples, channels]
            # If the shape is already [samples, channels], remove the .T
            print(f"Shape of stem from demix_track: {target_stem_np.shape}")
            target_stem_output = target_stem_np.T
            print(f"Shape after transpose: {target_stem_output.shape}")
            # --- END CORRECTION ---

            if original_mono:
                print("Converting target stem back to mono.")
                target_stem_output = target_stem_output[:, 0]

            # Calculate instrumental
            print("Calculating instrumental stem...")
            if original_mono:
                if len(original_mix.shape) != 1:
                     raise ValueError("Shape mismatch: original_mix is not mono while target_stem is.")
                instrumental_output = original_mix - target_stem_output
            else:
                if original_mix.shape != target_stem_output.shape:
                    # Attempt to fix length mismatch if minor (e.g., off-by-one from convolutions)
                    min_len = min(original_mix.shape[0], target_stem_output.shape[0])
                    print(f"Warning: Shape mismatch for instrumental calculation: Original {original_mix.shape}, Target {target_stem_output.shape}. Truncating to {min_len} samples.")
                    instrumental_output = original_mix[:min_len] - target_stem_output[:min_len]
                else:
                    instrumental_output = original_mix - target_stem_output
            print("Instrumental stem calculated.")

            # --- Save primary outputs to temporary files ---
            print("Saving primary output stems...")
            vocals_path_str = os.path.join(output_dir, "vocals.wav")
            instrumental_path_str = os.path.join(output_dir, "instrumental.wav")

            sf.write(vocals_path_str, target_stem_output, sr, subtype='FLOAT')
            sf.write(instrumental_path_str, instrumental_output, sr, subtype='FLOAT')
            print(f"Vocals saved to: {vocals_path_str}")
            print(f"Instrumental saved to: {instrumental_path_str}")

            # --- Resample Vocals to 16k Mono High quality for Whisper / Gladia ---
            vocals_16k_mono_path_str = os.path.join(output_dir, "vocals_16k_mono.wav")
            print(f"\nStarting high-quality resampling of vocals to 16k mono...")
            resample_to_16k_high_quality(vocals_path_str, vocals_16k_mono_path_str)
            print(f"Resampled vocals saved to: {vocals_16k_mono_path_str}\n")
            # --------------------------------------------

            end_time = time.time()
            print(f"Prediction finished in {end_time - start_time:.2f} seconds.")

            # Return paths to the saved files using cog.Path
            return Output(
                vocals=Path(vocals_path_str),
                instrumental=Path(instrumental_path_str),
                vocals_16k_mono=Path(vocals_16k_mono_path_str) # Added resampled path
            )

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            # Clean up temp dir on error? Maybe not, could be useful for debugging
            # if os.path.exists(output_dir):
            #     import shutil
            #     shutil.rmtree(output_dir)
            raise # Re-raise the exception so Cog marks the prediction as failed