# 方案调研：EML 符号回归 / Research: EML SR

## 全局方案（Search）/ Global options

| 候选 / Candidate | 类型 / Type | 特点 / Notes | Tradeoff | 决策 / Decision |
|------------------|-------------|--------------|----------|-----------------|
| PhySO | 库 / lib | Feynman 120 题 / 120 eqs | 非 EML 链 / not EML chain | adapt 公式来源 / formulas only |
| PySR / gplearn | 库 / lib | 遗传编程 / GP SR | 与线性链不兼容 / not linear chain | avoid |
| sympy+lambdify | 工具链 / stack | 符号+编译 / symbolic+compile | 自研树与坐标 / custom tree | adopt |

## 按抽象调研 / Per-abstraction

### 符号化简 / Symbolic simplify

| 候选 | 特点 | Tradeoff | 决策 |
|------|------|----------|------|
| sympy Function.fdiff | 硬编码 Tetration 导数 / hard-coded T fdiff | 自研合并 / custom merge | 自研 / build |
| sympy auto diff | exp 爆炸 / blow-up | 表达式膨胀 / huge exprs | avoid |

### 数值稳定 / Numeric stability

| 候选 | 特点 | Tradeoff | 决策 |
|------|------|----------|------|
| numpy clip | 简单 / simple | 非 C¹ / not C¹ | avoid |
| C¹ tail (§1.6) | 上下溢保护 / overflow guard | 复数实部处理 / complex Re | 自研 / build |

### Feynman 数据 / Data

| 候选 | 特点 | Tradeoff | 决策 |
|------|------|----------|------|
| PhySO CSV | 120 题完整 / full set | 额外依赖 / dep | wrap 思路 / idea only |
| 内置子集 / built-in | 轻量可测 / light | 覆盖有限 / limited | 自研 / build |

### 训练编排 / Training

| 候选 | 特点 | Tradeoff | 决策 |
|------|------|----------|------|
| 符号梯度+lambdify / sym grads | 与 DESIGN 一致 / matches design | 批处理需封装 / batch wrap | adopt |
| 手写链 autograd / hand chain | 快原型 / fast proto | 与符号 grad 不一致 / inconsistent | avoid |
