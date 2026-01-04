"""
Runtime environment utilities for optional dependency checking.

This module provides helpers to detect and validate optional dependencies
like flash-attn and xformers at runtime without failing on import.
"""

import importlib
import importlib.util
import logging
import warnings
from typing import Optional

logger = logging.getLogger("DZ-FaceDetailer.env")


def has_module(name: str) -> bool:
    """
    Check if a module is available without importing it.

    Args:
        name: The module name to check (e.g., "flash_attn", "xformers")

    Returns:
        True if the module can be found, False otherwise
    """
    spec = importlib.util.find_spec(name)
    return spec is not None


def require_module(name: str, hint: str = "") -> None:
    """
    Require a module to be installed, raising an error if missing.

    Args:
        name: The module name to require
        hint: Optional installation hint to show in the error message

    Raises:
        RuntimeError: If the module is not available
    """
    if not has_module(name):
        error_msg = f"Missing optional dependency: {name}"
        if hint:
            error_msg += f". {hint}"
        raise RuntimeError(error_msg)


def get_module_version(name: str) -> Optional[str]:
    """
    Get the version of an installed module.

    Args:
        name: The module name

    Returns:
        Version string if found, None otherwise
    """
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", None)
    except (ImportError, AttributeError):
        return None


def check_flash_attn_compatibility() -> bool:
    """
    Check if flash-attn is installed with a compatible version.

    Compatible versions: 2.7.1 to 2.8.2 (inclusive)

    Returns:
        True if compatible version is installed, False otherwise
    """
    if not has_module("flash_attn"):
        return False

    version_str = get_module_version("flash_attn")
    if not version_str:
        logger.warning("flash-attn installed but version cannot be determined")
        return False

    try:
        # Parse version (e.g., "2.8.2" -> (2, 8, 2))
        version_parts = tuple(int(x) for x in version_str.split(".")[:3])

        # Check if version is in range [2.7.1, 2.8.2]
        if (2, 7, 1) <= version_parts <= (2, 8, 2):
            logger.info("flash-attn %s detected (compatible)", version_str)
            return True
        else:
            logger.warning(
                "flash-attn %s detected but incompatible (requires >=2.7.1,<=2.8.2). "
                "Falling back to PyTorch attention.",
                version_str
            )
            return False

    except (ValueError, AttributeError) as e:
        logger.warning("Could not parse flash-attn version '%s': %s", version_str, e)
        return False


def check_xformers_availability() -> bool:
    """
    Check if xformers is available.

    Returns:
        True if xformers is installed, False otherwise
    """
    if has_module("xformers"):
        version = get_module_version("xformers")
        logger.info("xformers %s detected", version or "unknown version")
        return True

    return False


def log_runtime_environment() -> dict:
    """
    Log the runtime environment including optional dependencies.

    Returns:
        Dictionary with environment info
    """
    env_info = {
        "flash_attn_available": has_module("flash_attn"),
        "flash_attn_compatible": check_flash_attn_compatibility(),
        "xformers_available": check_xformers_availability(),
        "opencv_contrib_available": has_module("cv2"),
    }

    logger.info("Runtime environment: %s", env_info)
    return env_info
