# Vendored: PatchTST-FM

Source: [granite-tsfm](https://github.com/ibm-granite/granite-tsfm) 0.3.12 (PyPI wheel),
`tsfm_public/models/patchtst_fm/`. License: Apache-2.0 (`LICENSE` in this folder).

Why vendored: granite-tsfm requires `torch>=2.10,<2.12`, which conflicts with this project's
`torch==2.14.1`. The model files only need torch, transformers and einops.

Copied unchanged: `basic.py`, `configuration_patchtst_fm.py`, `normalization.py`.

Changed:

- `modeling_patchtst_fm.py`: `from .tools import count_parameters` replaced by an inlined copy of
  that function (`tools.py` is not vendored).
- `__init__.py`: written here; plain imports instead of the upstream lazy module.

To update, copy the same files from a newer granite-tsfm release and re-apply the change above.
