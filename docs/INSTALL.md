# Installation Guide

Python 3.10 – 3.12 (3.13 is not supported: the pinned `numpy<2` ships no
cp313 wheels).

## Install with pip

```bash
pip install .
pip install '.[dev]'  # also installs the development tools
```

Post-training additionally needs the LeRobot dataset tooling. `lerobot==0.3.3`
pins `torch<2.8`, which conflicts with this stack, so it must be installed
**without** its dependency pins — its actual runtime deps are already covered
by `requirements.txt`:

```bash
pip install -r requirements.txt
pip install lerobot==0.3.3 --no-deps
```

### Optional: `flash-attn`

`flash-attn` is only needed for `attn_mode='flashattn'` (serving uses SDPA and
training uses flex attention), so it is not a hard dependency. It cannot build
inside pip's PEP-517 isolation before torch exists — install it AFTER the steps
above:

#### No-build-isolation install (recommended)
```bash
pip install --upgrade pip setuptools wheel
pip install flash-attn --no-build-isolation
```

#### Install from git (alternative)
```bash
pip install git+https://github.com/Dao-AILab/flash-attention.git
```

---

### Next steps

Follow [DEPLOY.md](DEPLOY.md) to load the model and serve it, or
[POST_TRAINING.md](POST_TRAINING.md) to adapt it to your own data.

#### Formatting
```bash
black n0_twam
isort n0_twam
```
