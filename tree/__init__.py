"""Linear EML tree geometry and slot bindings. / 线性链几何与槽位绑定。"""
from .coords import (
    EXP_OFFSET,
    LEFT_OFFSET,
    LN_OFFSET,
    OUTPUT_COORD,
    RIGHT_OFFSET,
    X_SYMBOL,
    SlotBinding,
    c_to_d,
    d_to_c,
    layer_is_left,
    normalize_param,
    param_symbol,
    slot_bindings,
    slot_coords,
    input_slot_indices,
    learnable_slot_indices,
    x_slot_index,
)

__all__ = [
    'EXP_OFFSET', 'LEFT_OFFSET', 'LN_OFFSET', 'OUTPUT_COORD', 'RIGHT_OFFSET',
    'X_SYMBOL', 'SlotBinding', 'all_d_params', 'c_to_d', 'd_to_c', 'layer_is_left',
    'input_slot_indices', 'learnable_d_params', 'learnable_slot_indices',
    'normalize_param',
    'slot_bindings', 'slot_coords', 'x_slot_index',
]
