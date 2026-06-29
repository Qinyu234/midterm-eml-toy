# 架构拆分：EML 符号回归 / Architecture: EML SR

## 目录（按职责）/ Layout by responsibility

| 包 / Package | 职责 / Role | DESIGN |
|--------------|-------------|--------|
| `codec/` | 8-bit node 编解码 / node codec | §2.2, §5.1 |
| `tree/` | 线性链叶槽坐标、参数归一化 / slot coords & normalize | §1.7, §2.1 |
| `symbolic/` | sympy 构造 value/grad、Tetration、Arg / symbolic pipeline | §1.2–1.3 |
| `primitives/` | safe_ln、soft_exp 数值原语 / numeric guards | §1.5–1.6 |
| `export/` | 纯符号公式导出（拒绝数值）/ symbolic-only export | §2.3 |
| `compile/` | lambdify 编译 / compile to numpy | §1.2 |
| `numerics/` | 符号梯度训练模型 / training with sym grads | §3 |
| `data/`, `eval/`, `experiment/`, `strategy/` | 数据、评估、实验、策略桩 | §5–6 |
| `research/` | 未入主路径的研究工作区 / off-path research | §1.8 |
| `scripts/` | CLI 入口 / CLI entry points | — |

## 计算策略（§1.2）/ Compute strategy

```
symbolic/pipeline: build tree → simplify → value_expr; diff(cᵢ) → grad_exprᵢ
compile:           lambdify; ln→safe_ln(log1p); exp→soft_exp
numerics:          grad_exprᵢ + projection chain rule
```

## Node 编码（§2.2）

```python
low = byte & 0x0F
x_slot = (low >> 3) & 0x1
n_eml  = low & 0x7
reserved = (byte >> 4) & 0xF
```

遍历 `range(256)`；`n_eml == 0` 跳过。
