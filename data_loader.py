"""
Compatibility shim: root `train.py` and `experiments/*/train.py` import `data_loader`.

Implementation lives in `scripts/data_loader.py`.
"""

from scripts.data_loader import *  # noqa: F401,F403
