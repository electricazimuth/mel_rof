      
# Add this near the top imports in both files
import torch
import time
import inspect
import numpy as np
import gc # For garbage collection

# --- Memory Logging Utilities ---
def format_bytes(size):
    """Converts bytes to a human-readable format (KiB, MiB, GiB)."""
    if size == 0:
        return "0 B"
    power = 1024
    n = 0
    power_labels = {0: ' B', 1: ' KiB', 2: ' MiB', 3: ' GiB', 4: ' TiB'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f}{power_labels.get(n, '??')}"

def log_memory(label="", print_summary=False, device=None, do_output=False):
    """Logs current and peak PyTorch GPU memory usage."""
    if not do_output:
        return
    if not torch.cuda.is_available():
        # print(f"[{time.time():.2f} - MEM LOG @ {label}] CUDA not available.")
        return # Don't log if no CUDA

    if device is None:
        device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    # max_reserved = torch.cuda.max_memory_reserved(device) # Often less informative than max_allocated

    # Get caller function name for context
    try:
        caller_frame = inspect.currentframe().f_back
        caller_name = caller_frame.f_code.co_name
        line_no = caller_frame.f_lineno
        caller_info = f"{caller_name}:{line_no}"
    except Exception:
        caller_info = "Unknown Caller"


    print(
        f"[{time.time():.2f} - MEM LOG @ {caller_info} - {label}] -- "
        f"Alloc: {format_bytes(allocated)}, "
        f"Reserv: {format_bytes(reserved)}, "
        f"Peak Alloc: {format_bytes(max_allocated)}"
        # f"|| Peak Reserv: {format_bytes(max_reserved)}" # Optional
    )
    if print_summary:
         # Provides a detailed breakdown, can be very verbose
         print(torch.cuda.memory_summary(device, abbreviated=True))

def log_tensor_memory(tensor, name="Tensor", label="", do_output=False):
    """Logs shape, dtype, device, and estimated size of a tensor or numpy array."""
    if not do_output:
        return
    caller_frame = inspect.currentframe().f_back
    caller_name = caller_frame.f_code.co_name
    line_no = caller_frame.f_lineno
    caller_info = f"{caller_name}:{line_no}"
    prefix = f"[{time.time():.2f} - TENSOR LOG @ {caller_info} - {label}] -- '{name}' -- "

    if isinstance(tensor, torch.Tensor):
        size_bytes = tensor.nelement() * tensor.element_size() if tensor.numel() > 0 else 0
        print(f"{prefix} Type: torch.Tensor, Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}, Size: {format_bytes(size_bytes)}")
    elif isinstance(tensor, np.ndarray):
        size_bytes = tensor.nbytes
        print(f"{prefix} Type: np.ndarray, Shape: {tensor.shape}, Dtype: {tensor.dtype}, Size: {format_bytes(size_bytes)}")
    elif tensor is None:
         print(f"{prefix} Tensor is None")
    else:
        print(f"{prefix} Type: {type(tensor)}, Value: {str(tensor)[:100]}...") # Print limited value for other types

def trigger_gc(label=""):
    """Explicitly trigger garbage collection and log."""
    caller_frame = inspect.currentframe().f_back
    caller_name = caller_frame.f_code.co_name
    line_no = caller_frame.f_lineno
    caller_info = f"{caller_name}:{line_no}"
    print(f"[{time.time():.2f} - GC @ {caller_info} - {label}] Triggering garbage collection...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[{time.time():.2f} - GC @ {caller_info} - {label}] torch.cuda.empty_cache() called.")
    log_memory(f"After GC ({label})")

# --- End Memory Logging Utilities ---

    