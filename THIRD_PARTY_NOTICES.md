# Third-party notices

This repository contains code that was **not** written by the authors of MSE.
It consists of the **inference and evaluation** side of the project together
with the definitions of the backbone networks. The original MSE modules
(CPR / PDWL / SGM) live in the training loop, which is **not** shipped here;
only the CPR point refiner is kept because the localisation-error study
(`tools/make_cpr_simulation.py`) depends on it.

Each third-party component is listed below with its upstream source and licence.
Permissive licences (Apache-2.0 / MIT) require that the original copyright
notice and licence text are retained, which is done in the corresponding source
files.

---

## 1. Reused components

### ResNeSt — `model/LCAE/resnet2020.py`, `model/LCAE/splat.py`

- Upstream: <https://github.com/zhanghang1989/ResNeSt>
- Author: Hang Zhang and contributors
- Licence: **Apache-2.0**
- Status: verbatim copy. The original copyright header at the top of
  `resnet2020.py` has been deliberately preserved and must **not** be removed.
  Docstrings and inline comments elsewhere have been stripped; that is the only
  modification.

### DySample — `model/LCAE/DySample.py`

- Upstream: <https://github.com/tiny-smart/dysample>
- Paper: *DySample: Learning to Upsample by Learning to Sample* (ICCV 2023)
- Licence: **Apache-2.0** at the time of writing 
- Status: adapted (docstrings/comments removed, attribution line added).

### BASNet edge helpers — `components/edges.py`

- Upstream: <https://github.com/NathanUA/BASNet>
- Paper: *BASNet: Boundary-Aware Salient Object Detection* (CVPR 2019)
- Licence: **MIT**
- Status: adapted (`onehot_to_binary_edges` / `mask_to_onehot` are used to build
  the edge map consumed by PDWL).

### CBAM — `model/RDIAN/cbam.py`

- Upstream: <https://github.com/Jongchan/attention-module>
- Paper: *CBAM: Convolutional Block Attention Module* (ECCV 2018)
- Licence: **verify upstream** — the repository does not currently carry a
  standard permissive licence file.
- Status: adapted; used only by the RDIAN baseline.

---

## 2. Backbone networks

`model/ACM`, `model/ALC`, `model/DNA`, `model/GGL`, `model/LCAE`, `model/LWIRST`,
`model/MLCL`, `model/RDIAN`, `model/UIU` contain re-implementations of the
following **fully-supervised** segmentation backbones. In our paper each backbone
is trained under several settings — full supervision, and the weakly-supervised
frameworks we compare against and our own (**Ours / MSE**) —
so that the comparison tables can be reproduced. `LCAE` is the backbone used as
the default network in `test_model.py`.

| Directory | Backbone | Upstream |
| --- | --- | --- |
| `ACM` | Asymmetric Contextual Modulation for Infrared Small Target Detection (WACV 2021) | <https://github.com/YimianDai/open-acm> |
| `ALC` | Attention Local Contrast network | verify source repository |
| `DNA` | Deep Networks with Adaptive attention | verify source repository |
| `GGL` | Gradient-guided / global-local network | verify source repository |
| `LCAE` | Directional-contrast-gated / local-contrast attention encoder | verify source repository |
| `LWIRST` | Lightweight infrared small target detection | verify source repository |
| `MLCL` | Multi-level local contrast network | verify source repository |
| `RDIAN` | Robust Detection of Infrared small targets with Adaptive Network | verify source repository |
| `UIU` | Uncertainty-aware / UIU-Net style detector | <https://github.com/dafengshan/UIU-Net> |


