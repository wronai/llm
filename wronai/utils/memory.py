"""
Memory management utilities for WronAI.
"""

import gc
import psutil
import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional, Any

import torch

from .logging import get_logger

logger = get_logger(__name__)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory usage statistics
    """
    memory_info = {}

    # CPU memory
    cpu_memory = psutil.virtual_memory()
    memory_info.update({
        "cpu_memory_total_gb": cpu_memory.total / (1024 ** 3),
        "cpu_memory_used_gb": cpu_memory.used / (1024 ** 3),
        "cpu_memory_percent": cpu_memory.percent
    })

    # GPU memory
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            memory_used = torch.cuda.memory_allocated() / (1024 ** 3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            memory_percent = (memory_used / memory_total) * 100

            memory_info.update({
                "gpu_memory_total_gb": memory_total,
                "gpu_memory_used_gb": memory_used,
                "gpu_memory_percent": memory_percent,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
                "gpu_memory_cached_gb": torch.cuda.memory_cached() / (1024 ** 3)
            })
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            memory_info.update({
                "gpu_memory_total_gb": 0,
                "gpu_memory_used_gb": 0,
                "gpu_memory_percent": 0
            })
    else:
        memory_info.update({
            "gpu_memory_total_gb": 0,
            "gpu_memory_used_gb": 0,
            "gpu_memory_percent": 0
        })

    return memory_info


def clear_cache():
    """Clear GPU and CPU caches."""
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Clear CPU cache
    gc.collect()

    logger.debug("Memory caches cleared")


def format_memory_usage(memory_info: Dict[str, float]) -> str:
    """
    Format memory usage information as human-readable string.

    Args:
        memory_info: Memory usage dictionary

    Returns:
        Formatted string
    """
    lines = []

    # CPU memory
    cpu_used = memory_info.get("cpu_memory_used_gb", 0)
    cpu_total = memory_info.get("cpu_memory_total_gb", 0)
    cpu_percent = memory_info.get("cpu_memory_percent", 0)
    lines.append(f"CPU: {cpu_used:.1f}GB / {cpu_total:.1f}GB ({cpu_percent:.1f}%)")

    # GPU memory
    gpu_used = memory_info.get("gpu_memory_used_gb", 0)
    gpu_total = memory_info.get("gpu_memory_total_gb", 0)
    gpu_percent = memory_info.get("gpu_memory_percent", 0)

    if gpu_total > 0:
        lines.append(f"GPU: {gpu_used:.1f}GB / {gpu_total:.1f}GB ({gpu_percent:.1f}%)")
    else:
        lines.append("GPU: Not available")

    return " | ".join(lines)


@contextmanager
def memory_monitor(interval: float = 1.0, log_peak: bool = True):
    """
    Context manager to monitor memory usage during execution.

    Args:
        interval: Monitoring interval in seconds
        log_peak: Whether to log peak memory usage

    Yields:
        MemoryMonitor instance
    """
    monitor = MemoryMonitor(interval=interval)
    monitor.start()

    try:
        yield monitor
    finally:
        monitor.stop()

        if log_peak:
            peak_info = monitor.get_peak_usage()
            logger.info(f"Peak memory usage: {format_memory_usage(peak_info)}")


class MemoryMonitor:
    """
    Real-time memory usage monitor.
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.thread = None
        self.memory_history = []
        self.peak_usage = {}

    def start(self):
        """Start memory monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.debug("Memory monitoring started")

    def stop(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.debug("Memory monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                memory_info = get_memory_usage()
                timestamp = time.time()

                # Add timestamp to memory info
                memory_info["timestamp"] = timestamp

                # Store in history
                self.memory_history.append(memory_info)

                # Update peak usage
                self._update_peak_usage(memory_info)

                # Keep only last 1000 entries to prevent memory leak
                if len(self.memory_history) > 1000:
                    self.memory_history = self.memory_history[-1000:]

                time.sleep(self.interval)

            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                time.sleep(self.interval)

    def _update_peak_usage(self, current_usage: Dict[str, float]):
        """Update peak memory usage."""
        for key, value in current_usage.items():
            if key != "timestamp" and isinstance(value, (int, float)):
                if key not in self.peak_usage or value > self.peak_usage[key]:
                    self.peak_usage[key] = value

    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        return get_memory_usage()

    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak memory usage since monitoring started."""
        return self.peak_usage.copy()

    def get_history(self, last_n: Optional[int] = None) -> list:
        """
        Get memory usage history.

        Args:
            last_n: Number of last entries to return

        Returns:
            List of memory usage entries
        """
        if last_n is None:
            return self.memory_history.copy()
        else:
            return self.memory_history[-last_n:].copy()

    def get_average_usage(self, window_size: Optional[int] = None) -> Dict[str, float]:
        """
        Get average memory usage over specified window.

        Args:
            window_size: Number of entries to average over

        Returns:
            Average memory usage
        """
        history = self.get_history(window_size)

        if not history:
            return {}

        # Calculate averages
        avg_usage = {}
        for key in history[0].keys():
            if key != "timestamp" and isinstance(history[0][key], (int, float)):
                avg_usage[key] = sum(entry[key] for entry in history) / len(history)

        return avg_usage


def estimate_model_memory(
        num_parameters: int,
        precision: str = "float16",
        quantization: Optional[str] = None,
        batch_size: int = 1,
        sequence_length: int = 2048
) -> Dict[str, float]:
    """
    Estimate memory requirements for a model.

    Args:
        num_parameters: Number of model parameters
        precision: Model precision (float32, float16, bfloat16)
        quantization: Quantization type (4bit, 8bit)
        batch_size: Batch size for inference/training
        sequence_length: Sequence length

    Returns:
        Memory estimation in GB
    """
    # Bytes per parameter based on precision
    precision_bytes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 0.5
    }

    # Determine effective precision
    if quantization == "4bit":
        bytes_per_param = precision_bytes["int4"]
    elif quantization == "8bit":
        bytes_per_param = precision_bytes["int8"]
    else:
        bytes_per_param = precision_bytes.get(precision, 2)

    # Model weights
    model_memory = num_parameters * bytes_per_param / (1024 ** 3)

    # Activation memory (rough estimate)
    # Assuming transformer with hidden_size proportional to sqrt(num_parameters)
    hidden_size = int((num_parameters / 1000000) ** 0.5 * 1000)  # Rough estimate
    activation_memory = batch_size * sequence_length * hidden_size * 4 / (1024 ** 3)

    # Gradient memory (for training)
    gradient_memory = model_memory if quantization is None else 0

    # Optimizer states (for training with Adam)
    optimizer_memory = model_memory * 2 if quantization is None else 0

    return {
        "model_memory_gb": model_memory,
        "activation_memory_gb": activation_memory,
        "gradient_memory_gb": gradient_memory,
        "optimizer_memory_gb": optimizer_memory,
        "total_inference_gb": model_memory + activation_memory,
        "total_training_gb": model_memory + activation_memory + gradient_memory + optimizer_memory
    }


def optimize_memory_usage(
        model: torch.nn.Module,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision: bool = True,
        clear_cache_frequency: int = 100
):
    """
    Apply memory optimization techniques to a model.

    Args:
        model: PyTorch model to optimize
        enable_gradient_checkpointing: Enable gradient checkpointing
        enable_mixed_precision: Enable mixed precision training
        clear_cache_frequency: How often to clear cache (in steps)
    """
    optimizations_applied = []

    # Gradient checkpointing
    if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        optimizations_applied.append("gradient_checkpointing")

    # Mixed precision is handled by training arguments, just log
    if enable_mixed_precision:
        optimizations_applied.append("mixed_precision")

    logger.info(f"Memory optimizations applied: {', '.join(optimizations_applied)}")

    return optimizations_applied


class MemoryProfiler:
    """
    Detailed memory profiler for debugging memory issues.
    """

    def __init__(self):
        self.profiles = []
        self.baseline = None

    def set_baseline(self):
        """Set memory baseline."""
        self.baseline = get_memory_usage()
        logger.info(f"Memory baseline set: {format_memory_usage(self.baseline)}")

    def profile(self, name: str):
        """
        Create a memory profile snapshot.

        Args:
            name: Name for this profile point
        """
        current_usage = get_memory_usage()

        profile_data = {
            "name": name,
            "timestamp": time.time(),
            "usage": current_usage
        }

        # Calculate delta from baseline
        if self.baseline:
            delta = {}
            for key, value in current_usage.items():
                baseline_value = self.baseline.get(key, 0)
                delta[f"delta_{key}"] = value - baseline_value
            profile_data["delta"] = delta

        self.profiles.append(profile_data)

        logger.debug(f"Memory profile '{name}': {format_memory_usage(current_usage)}")

    def get_summary(self) -> Dict[str, Any]:
        """Get memory profiling summary."""
        if not self.profiles:
            return {"error": "No profiles recorded"}

        summary = {
            "num_profiles": len(self.profiles),
            "baseline": self.baseline,
            "profiles": self.profiles,
            "peak_usage": {}
        }

        # Find peak usage across all profiles
        for profile in self.profiles:
            for key, value in profile["usage"].items():
                if isinstance(value, (int, float)):
                    current_peak = summary["peak_usage"].get(key, 0)
                    summary["peak_usage"][key] = max(current_peak, value)

        return summary

    def print_summary(self):
        """Print memory profiling summary."""
        summary = self.get_summary()

        print("\n=== Memory Profiling Summary ===")
        print(f"Number of profiles: {summary['num_profiles']}")

        if summary.get("baseline"):
            print(f"Baseline: {format_memory_usage(summary['baseline'])}")

        if summary.get("peak_usage"):
            print(f"Peak usage: {format_memory_usage(summary['peak_usage'])}")

        print("\nProfile timeline:")
        for i, profile in enumerate(summary["profiles"]):
            name = profile["name"]
            usage = format_memory_usage(profile["usage"])
            print(f"  {i + 1}. {name}: {usage}")

            if profile.get("delta"):
                gpu_delta = profile["delta"].get("delta_gpu_memory_used_gb", 0)
                cpu_delta = profile["delta"].get("delta_cpu_memory_used_gb", 0)
                if abs(gpu_delta) > 0.1 or abs(cpu_delta) > 0.1:
                    print(f"     Delta: GPU {gpu_delta:+.1f}GB, CPU {cpu_delta:+.1f}GB")


def check_memory_requirements(
        required_memory_gb: float,
        safety_margin: float = 0.1
) -> bool:
    """
    Check if system has enough memory for operation.

    Args:
        required_memory_gb: Required memory in GB
        safety_margin: Safety margin (0.1 = 10%)

    Returns:
        True if sufficient memory is available
    """
    current_usage = get_memory_usage()

    # Check GPU memory if available
    if torch.cuda.is_available():
        gpu_total = current_usage.get("gpu_memory_total_gb", 0)
        gpu_used = current_usage.get("gpu_memory_used_gb", 0)
        gpu_available = gpu_total - gpu_used

        required_with_margin = required_memory_gb * (1 + safety_margin)

        if gpu_available >= required_with_margin:
            return True
        else:
            logger.warning(
                f"Insufficient GPU memory: {gpu_available:.1f}GB available, "
                f"{required_with_margin:.1f}GB required"
            )
            return False

    # Fallback to CPU memory check
    cpu_total = current_usage.get("cpu_memory_total_gb", 0)
    cpu_used = current_usage.get("cpu_memory_used_gb", 0)
    cpu_available = cpu_total - cpu_used

    required_with_margin = required_memory_gb * (1 + safety_margin)

    if cpu_available >= required_with_margin:
        return True
    else:
        logger.warning(
            f"Insufficient CPU memory: {cpu_available:.1f}GB available, "
            f"{required_with_margin:.1f}GB required"
        )
        return False