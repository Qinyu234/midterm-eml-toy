# eML — 实验玩具（已归档）

[English](README.en.md) · **简体中文** · [Index](README.md)

> 期中考试无聊 + 额度没用完 → 随便整了点玩具。

**状态：** 归档 · 负结果 · 不再维护

---

## 这是什么

用固定算子 `eml(z,w) = exp(z) - ln(w)` 搭了一棵 **8-bit 编码的 tetation 树**，在复数叶槽 `D` 上做符号求导 + 数值拟合，和极小 MLP / Fourier 基线比 1D 回归。

包含：符号 pipeline、16-node sweep、SCORE benchmark、Feynman 梯度矩阵、MLP 权重列探针等。

---

## 本来打算 vs 实际交付

**原计划（本地跑不动，未完整实现）：**

- **256 个 8-bit node** 全 sweep（`n_eml` × `x_slot` × 高 4 bit `reserved`）
- **多 node 叠加 / 串联**：用高 nibble 把多棵 eml 树串成更深组合，做 node×node 结构搜索
- **node × Feynman 全矩阵** + 完整 SCORE 级 benchmark，本地单机内存/时间扛不住（符号化简膨胀、进程被 kill、单次全量 SCORE 数小时）

**仓库里实际有的：**

- 单棵树的 **16 低 nibble sweep**（14 个可 eval，`n_eml=0` 跳过）
- 高 nibble / **node 叠加** 仅在编码与文档层预留（`codec/node.py`），**无串联求值与训练**
- 可本地跑通的：`pytest`、smoke SCORE、全量 SCORE（15 targets，约 40 分钟量级，已测）、MLP 列探针

若你看到 `range(256)`、`多 node 串联` 等字样，那是设计草稿，**不是本仓库可复现路径**。

---

## 未完想法（记录即归档）

以下有设计、部分有草稿或局部代码，**没跑通 / 没进主路径**，不再做。

| 方向 | 想法 | 状态 |
|------|------|------|
| **Node 叠加嵌套** | 低 nibble 单 node sweep 跑完后，用高 nibble 把多棵 eml 树嵌套组合 | 本地跑不动，**放弃** |
| **`log1p` / `expm1`** | 数值层用稳定初等函数替代裸 `ln(1+z)`、`exp(z)-1`，减轻小量区梯度病态 | 有想法；`lambdify` 表里有映射，**未全面接入符号化简与训练** |
| **Logsum 推广** | `T(e, -1, T(e, n, z) + T(e, n, w))` 的稳定求值与合并（gap 阈值、成对 ln-sum） | 部分落在 `symbolic/logsum_gap.py`（SCORE 前修过 overflow），**推广规则与 optimize 管线未做完** |
| **256 全 sweep** | node×Feynman 全矩阵、叠加搜索 | 见上，**未实现** |

代码里若见到相关 stub 或半完成模块，当作**历史草稿**，不是承诺功能。

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
