# 任务清单：EML 符号回归 / Task List: EML SR

> 顺序 / Order：Tasks → Tests → Code（见 coding-workflow skill）

## 验收 / Acceptance

- [x] 全部 pytest 通过 / all pytest pass
- [x] `scripts/export_symbols.py` 可导出符号结果 / phase1 symbol export

## 任务 / Tasks

| ID | 策略 / Strategy | 任务 / Task | 交付 / Deliverable | Tests |
|----|-----------------|-------------|-------------------|-------|
| T1 | DRY | codec + tree + primitives + symbolic | `codec/`, `tree/`, `primitives/`, `symbolic/` | `test_node.py`, `test_operator.py` |
| T2 | Curry | symbolic.pipeline + lambdify | `symbolic/pipeline.py`, `compile/` | `test_phase1_symbolic.py`, `test_compile.py` |
| T3 | Curry | 符号梯度模型 + 投影 / symbolic-grad model | `numerics/` | `test_chain.py` |
| T4 | DRY | Feynman 数据 + 采样 / data + sampling | `data/` | `test_data.py` |
| T5 | Meta | Node 评估 + cache + 实验 / eval + experiment | `eval/`, `experiment/` | `test_node_eval.py`, … |
| T6 | Meta | 策略 G₀/G₁/G₂ 桩 / strategy stubs | `strategy/` | `test_strategy.py` |

## 实现顺序 / Implementation order

1. T1 — tests → code  
2. T2 — symbolic + compile  
3. T3 — symbolic_model  
4. T4 — data  
5. T5 — eval  
6. T6 — strategy stubs  
