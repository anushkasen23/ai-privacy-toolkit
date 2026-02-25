"""
The AI Privacy Toolbox (apt).
"""

try:  # pragma: no cover - optional dependency chains
    from apt import anonymization
except Exception:  # pragma: no cover
    anonymization = None

try:  # pragma: no cover - optional dependency chains
    from apt import minimization
except Exception:  # pragma: no cover
    minimization = None

try:  # pragma: no cover - optional dependency chains
    from apt import utils
except Exception:  # pragma: no cover
    utils = None

__version__ = "0.2.1"
