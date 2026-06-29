# eML — experimental toy (archived)

[English](README.en.md) · [简体中文](README.zh.md) · [Index](README.md)

> Midterm boredom + unused compute quota → a small experimental toy.

**Status:** archived · negative results · not maintained

---

## What this is

An **8-bit encoded tetation tree** with fixed operator `eml(z,w) = exp(z) - ln(w)`, Wirtinger grads on complex leaf slots `D`, compared against tiny MLP / Fourier baselines on 1D regression.

Includes: symbolic pipeline, 16-node sweep, SCORE benchmark, Feynman grad matrix, MLP weight-column probe, etc.

---

## Conclusion (TL;DR)

**Under this implementation and evaluation, eML did not show enough value to keep investing.**

| Metric | eML | MLP / Fourier |
|--------|-----|---------------|
| SCORE eps=1e-2 pass | **0 / 210** | most pass |
| best in_rmse vs baselines | often **10²–10³×** worse | see `results/SCORE.txt` |
| MLP column / neuron probe | poor fit, unreadable formulas | — |

See `results/SCORE.txt`, `results/MLP_COLUMN*.txt`.

---

## Optimization bug or complex distortion?

**Both matter, but neither is the full story.**

### Optimization (partly yes)

- Only **leaf D + α/β** are tuned; **byte structure is fixed** — no real structure search.
- Symbolic grads + simple descent; tried complex MSE, LR decay, early stop — **still no eps pass**.
- MLP (AdamW/BFGS to convergence) passes under similar budgets → **not just “too few steps”**.

### Complex numbers (adds numeric risk)

- Real targets go through `Re/Im` projection or `|z-y|²`; extra DOF, **Im(z) can drift**.
- `exp`/`ln` on ℂ are fragile; `logsum_gap` fixed some overflows; **n_eml=2 still blows up** (see SCORE).
- But: the **hypothesis class is an exp/ln tower** — it does not naturally express `sin`/`tanh`/MLP weights.

**Verdict:** Better optimization might help marginally; complex training is brittle; root cause is **structure–target mismatch**. A restart should change **structure search / real-only branch / symbolic recovery metrics**, not just LR.

---

## Reproduce

**Python 3.11** (see `.python-version` in repo root)

```bash
cd py/eml
pip install -r requirements.txt
pytest tests/ -q
python scripts/run_score.py --targets sin --max-steps 100
python scripts/run_score.py --targets all --max-steps 500   # full run, slow
python scripts/run_mlp_column.py --kind neuron_act --neuron 0
```

Docker (optional):

```bash
docker compose run --rm test
docker compose run --rm eml
```

Main outputs:

- `results/SCORE.txt` — 15 targets × (MLP + Fourier + 14 eML nodes)
- `results/MLP_COLUMN.txt` — tiny public MLP weight-column probe
- `FORMULA_WORKPLACE.txt` — symbolic formula export

---

## Layout

| Path | Description |
|------|-------------|
| `symbolic/` | sympy pipeline, tetration, Wirtinger, `logsum_gap` |
| `numerics/` | `CompiledEMLModel`, D binding |
| `tree/coords.py` | leaf coords, `bottom_d` input binding |
| `experiment/` | SCORE, search, TES, MLP column probe |
| `baselines/` | MLP / Fourier (AdamW, BFGS) |
| `data/targets.py` | classic + rule targets |
| `scripts/` | `run_score.py`, `run_mlp_column.py`, etc. |

Historical design docs: `PURPOSE.md`, `ARCHITECTURE.md`, `TASKS.md`

---

## License & disclaimer

Toy project, as-is. Fork for fun; **no warranty, no maintenance**.

---

*Written after full SCORE and MLP-column probes.*
