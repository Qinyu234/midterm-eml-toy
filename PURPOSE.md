# 任务目的：EML 符号回归实验程序 / Purpose: EML SR Program

## 要解决什么问题 / Problem

在固定 EML 算子 `eml(z,w)=exp(z)-ln(w)` 下，实现复数域符号回归管线，遍历 256 个 8-bit node 编码，评估各结构对 Feynman 目标函数的适配性。

Implement complex-domain SR with fixed EML operator; sweep 256 8-bit nodes; measure adaptability on Feynman targets.

## 为什么现在做 / Why now

DESIGN.md 已定义符号/数值边界、坐标归一化、Tetration 与评估指标；需要可运行程序验证设计并产出 node×function 矩阵。

DESIGN specifies symbolic/numeric split, normalization, Tetration, metrics; need runnable code and adaptability matrix.

## 验收标准 / Acceptance

- [ ] 符号 pipeline：value_expr + grad_expr 化简通过 / symbolic value + grads simplify
- [ ] 数值层：safe_ln(log1p) / soft_exp + lambdify / numeric guards via lambdify
- [ ] 数据：Feynman 子集 n_vars∈{1,2}，[-3,3] 采样 / Feynman subset, sampling
- [ ] 评估：有效 node 遍历，收敛性/RMSE/矩阵 / node sweep, metrics
- [ ] 全部 pytest 通过 / all pytest green

## 明确不做 / Out of scope

- G₀/G₁/G₂ 完整对比（仅桩）/ full strategy comparison (stubs only)
- Projection B/C；ln(T+T) 精确式；多 node 串联；硬化完整实现 / projection B/C, ln(T+T), multi-node, full snapping

## Grill 记录 / Grill log

| 问题 / Q | 结论 / A |
|----------|----------|
| 当前阶段核心目标？/ Phase goal? | 遍历 node 适配性（§5），非策略对比 / adaptability sweep, not G compare |
| 符号/数值边界？/ Boundary? | sympy 标准 exp/ln；lambdify→safe_ln/soft_exp / standard symbolic, guarded numeric |
| 输入 x？/ Input x? | x_slot 定槽位；`x` 在 value_expr 中 / x_slot + symbol x in value_expr |
| Node 编码？/ Encoding? | 8-bit；低 4 bit：x_slot+n_eml；高 4 bit 保留 / 8-bit, low nibble effective |
