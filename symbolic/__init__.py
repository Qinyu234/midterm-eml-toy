"""SymPy symbolic layer: pipeline, Tetration, Arg. / 符号层。"""
from .arg import ComplexArg, complex_log_symbolic
from .pipeline import (
    NodeSymbolicExprs,
    ParamGradient,
    build_value_tree,
    compute_node_exprs,
    eml,
    simplify_expr,
)
from .logsum_gap import (
    DEFAULT_GAP_THRESHOLD,
    LogsumT,
    gap_correction_symbolic,
    gap_threshold_for_dtype,
    gap_threshold_from_epsilon,
    ln_t_sum_symbolic,
    logsum_pair_symbolic,
    merge_t_exp_add,
    stabilize_ln_t_sum,
    t_preimage_symbolic,
)
from .optimize import (
    expand_ln_exp_algebra,
    fold_coeff_into_exp,
    log_div_exponent,
    log_div_value,
    merge_nested_tetration,
    optimize_symbolic,
    stabilize_exp_quotient,
)
from .wirtinger import (
    WirtingerJacobian,
    format_abi,
    simplify_wirtinger,
    wirtinger_grad,
)
from .tetration import (
    Tetration,
    compressed_exp,
    compressed_ln,
    expand_t_to_log_exp,
    format_t_expr,
    merge_tetration,
    merge_opposite_tetration,
    simplify_in_t,
    to_t_form,
)

__all__ = [
    'ComplexArg', 'complex_log_symbolic',
    'NodeSymbolicExprs', 'ParamGradient',
    'build_value_tree', 'compute_node_exprs', 'eml', 'simplify_expr',
    'merge_nested_tetration', 'optimize_symbolic', 'stabilize_exp_quotient',
    'fold_coeff_into_exp', 'expand_ln_exp_algebra', 'log_div_exponent', 'log_div_value',
    'E_TAG', 'Tetration', 'compressed_exp', 'compressed_ln', 'merge_tetration',
    'to_t_form', 'simplify_in_t', 'expand_t_to_log_exp', 'format_t_expr',
    'merge_opposite_tetration',
    'DEFAULT_GAP_THRESHOLD', 'LogsumT', 'gap_correction_symbolic',
    'gap_threshold_for_dtype', 'gap_threshold_from_epsilon',
    'ln_t_sum_symbolic', 'logsum_pair_symbolic', 'merge_t_exp_add',
    'stabilize_ln_t_sum', 't_preimage_symbolic',
    'WirtingerJacobian', 'wirtinger_grad', 'format_abi', 'simplify_wirtinger',
]
