"""Version information for WronAI."""

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

# Build information
__build__ = "alpha"
__commit__ = "main"
__date__ = "2025-06-13"

# Compatibility
MIN_PYTHON_VERSION = (3, 8)
MIN_TORCH_VERSION = "2.0.0"
MIN_TRANSFORMERS_VERSION = "4.35.0"


def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "build": __build__,
        "commit": __commit__,
        "date": __date__,
        "python_required": ".".join(map(str, MIN_PYTHON_VERSION)),
        "torch_required": MIN_TORCH_VERSION,
        "transformers_required": MIN_TRANSFORMERS_VERSION,
    }
