"""PatchTST-FM, vendored from IBM granite-tsfm 0.3.12 (`tsfm_public/models/patchtst_fm`), Apache-2.0.

Vendored because granite-tsfm pins torch<2.12 and this project pins torch 2.14.1; the model code
itself needs only torch, transformers and einops. See README.md here for what was changed.
"""

from .configuration_patchtst_fm import PatchTSTFMConfig
from .modeling_patchtst_fm import PatchTSTFMForPrediction, PatchTSTFMModel

__all__ = ["PatchTSTFMConfig", "PatchTSTFMForPrediction", "PatchTSTFMModel"]
