"""Baseline fitters: MLP, Fourier."""
from .base import FitResult
from .fourier import fit_fourier, scan_fourier
from .mlp import fit_mlp, scan_mlp

__all__ = [
    'FitResult',
    'fit_fourier', 'fit_mlp',
    'scan_fourier', 'scan_mlp',
]
