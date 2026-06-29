# eML — 实验玩具（已归档）

[English](README.en.md) · **简体中文** · [Index](README.md)

> 期中考试无聊 + 额度没用完 → 随便整了点玩具。

**状态：** 归档 · 负结果 · 不再维护

---

## 这是什么

用固定算子 `eml(z,w) = exp(z) - ln(w)` 搭了一棵 **8-bit 编码的 tetation 树**，在复数叶槽 `D` 上做符号求导 + 数值拟合，和极小 MLP / Fourier 基线比 1D 回归。

包含：符号 pipeline、16-node sweep、SCORE benchmark、Feynman 梯度矩阵、MLP 权重列探针等。

---

## 结论（先说）

**在当前实现与评测下，eML 没有表现出值得继续投入的价值。**

| 指标 | eML | MLP / Fourier |
|------|-----|---------------|
| SCORE eps=1e-2 pass | **0 / 210** | 多数通过 |
| best in_rmse vs baselines | 通常差 **10²–10³×** | 见 `results/SCORE.txt` |
| MLP 列 / neuron 探针 | 拟合差，公式不可读 | — |

详见 `results/SCORE.txt`、`results/MLP_COLUMN*.txt`。

---

## 是优化问题，还是复数失真？

**两者都有影响，但都不是主因。**

### 优化（有一部分）

- 只调 **叶槽 D + α/β**，**byte 结构固定**，没有真正的结构搜索。
- 符号梯度 + 简单下降；曾试复数 MSE、lr 衰减、早停，**仍未过 eps**。
- 同预算下 MLP（AdamW/BFGS 训到收敛）能过线 → **不全是「步数不够」**。

### 复数（加重数值风险）

- 实值目标要经过 `Re/Im` 投影或 `|z-y|²`，多自由度，**Im(z) 可乱跑**。
- `exp`/`ln` 在复平面敏感；`logsum_gap` 修了一部分 overflow，**n_eml=2 仍偶发爆炸**（见 SCORE）。
- 但：**表达集本身是 exp/ln 塔**，本来就不会自然长出 `sin`/`tanh`/MLP 权重形状。

**判断：** 换更好的优化可能会略好，但很难翻盘；复数让训练更脆，但根因是 **结构 + 目标函数不匹配**。若重开，更该动 **结构搜索 / 实数专用分支 / 符号恢复指标**，而不是再拧学习率。

---

## 复现

**Python 3.11**（见仓库 `.python-version`）

```bash
cd py/eml
pip install -r requirements.txt
pytest tests/ -q
python scripts/run_score.py --targets sin --max-steps 100
python scripts/run_score.py --targets all --max-steps 500   # 全量，慢
python scripts/run_mlp_column.py --kind neuron_act --neuron 0
```

Docker（可选）:

```bash
docker compose run --rm test
docker compose run --rm eml
```

主要输出：

- `results/SCORE.txt` — 15 targets × (MLP + Fourier + 14 eML nodes)
- `results/MLP_COLUMN.txt` — 公开 tiny MLP 权重列探针
- `FORMULA_WORKPLACE.txt` — 符号公式工作区导出

---

## 目录速写

| 路径 | 说明 |
|------|------|
| `symbolic/` | sympy pipeline、tetration、Wirtinger、`logsum_gap` |
| `numerics/` | `CompiledEMLModel`、D 绑定 |
| `tree/coords.py` | 叶槽坐标、`bottom_d` 输入绑定 |
| `experiment/` | SCORE、search、TES、MLP 列探针 |
| `baselines/` | MLP / Fourier（AdamW、BFGS） |
| `data/targets.py` | classic + rule 目标函数 |
| `scripts/` | `run_score.py`、`run_mlp_column.py` 等 |

设计文档（历史）：`PURPOSE.md`、`ARCHITECTURE.md`、`TASKS.md`

---

## 许可与免责

玩具项目，按现状提供（as-is）。欢迎 fork 玩，**不保证正确性、不维护**。

---

*写于全量 SCORE 与 MLP 列探针之后。*
