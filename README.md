# MSE

A weakly-supervised framework for
single-frame infrared small target detection (IRSTD) trained from **point
annotations only**.

> **Availability.** This repository currently ships only the **inference and
> evaluation code** and **checkpoint** so that the reported numbers can be
> verified. The **training code will be released after the paper is accepted**.

This repository contains the inference and evaluation code together with the
model definitions. The training loop is intentionally not included, so the three
modules that constitute the MSE contribution (CPR, PDWL, SGM) are **not** shipped
here — only the CPR point refiner is kept because the localisation-error study
depends on it. 

---

## Inference and evaluation

### Checkpoint

Download the pre-trained checkpoints from:

- **Quark netdisk:** <https://pan.quark.cn/s/381b1a29a4bb>
- **Extraction code:** `7JXr`

Each checkpoint is named
`{backbone}__{dataset}__{annotation}__{timestamp}_best.pth.tar` and corresponds
to our **Ours (MSE)** method trained on one backbone / dataset / annotation
combination.
Point `test_model_path` in `test_model.py` at the downloaded file.

### Running inference

Edit `choose_model`, `choose_dataset` and `test_model_path` at the top of
`test_model.py`, then:

```bash
python test_model.py
```

## CPR localisation-error study

The localisation error reported for CPR is measured on synthetic patches where
the target centre is known analytically — real annotated masks give no
sub-pixel reference point. The script is self-contained and imports the released
refiner, so the numbers are reproducible from the public code:

```bash
python tools/make_cpr_simulation.py --n 200
python tools/make_cpr_simulation.py --n 500 --seed 7 --json cpr_sim.json --csv cpr_sim.csv
python tools/make_cpr_simulation.py --no-clutter     # disable background clutter
```

## Acknowledgements

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the origin and licence
of every reused third-party implementation.

## Licence

TODO — choose a licence before publishing (see the `LICENSE` file in this
repository).
