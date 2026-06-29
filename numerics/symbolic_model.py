"""
EML model from symbolic simplify + lambdify (DESIGN §1.2).
All leaf slots are D_1..D_{n+1}; Wirtinger grads for each.
Feynman inputs bind to bottom |d| leaf D slots; rest are learnable.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from compile.lambdify_exprs import compile_node_exprs
from numerics.d_binding import bind_d_args
from symbolic.pipeline import compute_node_exprs
from tree.coords import BindingMode, input_slot_indices, learnable_slot_indices


class CompiledEMLModel(nn.Module):
    def __init__(
        self,
        byte: int,
        *,
        n_vars: int = 1,
        binding: BindingMode = 'bottom_d',
    ):
        super().__init__()
        self.byte = byte
        self.n_vars = n_vars
        self.binding = binding
        self.exprs = compute_node_exprs(byte)
        self.value_fn, self.grad_fns = compile_node_exprs(self.exprs)
        node = self.exprs.node
        self._input_slots = input_slot_indices(
            node.n_eml, node.x_slot, n_vars, binding=binding,
        )
        self._learnable_slots = learnable_slot_indices(
            node.n_eml, node.x_slot, n_vars, binding=binding,
        )
        self.d_params = nn.ParameterList([
            nn.Parameter(torch.randn((), dtype=torch.float64))
            for _ in self._learnable_slots
        ])
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def _learnable_numpy(self) -> List[float]:
        return [float(p.item()) for p in self.d_params]

    def _full_learnable_vector(self) -> List[float]:
        """n_eml+1 entries; learnable slots filled, others 0 (overwritten by inputs)."""
        vec = [0.0] * (self.exprs.node.n_eml + 1)
        for slot, val in zip(self._learnable_slots, self._learnable_numpy()):
            vec[slot] = val
        return vec

    def _bind_d_args(self, X: np.ndarray) -> Tuple[np.ndarray, ...]:
        return bind_d_args(
            self.exprs.node.n_eml,
            self.exprs.node.x_slot,
            self.n_vars,
            X,
            self._full_learnable_vector(),
            binding=self.binding,
        )

    def complex_output(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = self.value_fn(*self._bind_d_args(X))
        return np.asarray(out, dtype=np.complex128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy().astype(np.float64)
        z = self.complex_output(x_np)
        z_t = torch.tensor(z, dtype=torch.complex128, device=x.device)
        return (self.alpha * z_t.real + self.beta * z_t.imag).to(dtype=torch.float64)

    def loss_and_symbolic_grads(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        loss_mode: str = 'complex',
    ) -> Tuple[float, List[float], float, float]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        d_args = self._bind_d_args(X)
        alpha = float(self.alpha.item())
        beta = float(self.beta.item())

        z = np.asarray(self.value_fn(*d_args), dtype=np.complex128)
        if loss_mode == 'complex':
            y_c = y.astype(np.complex128)
            diff = z - y_c
            loss = float(np.mean(np.abs(diff) ** 2))
            scale = 2.0 / len(y)
            dL_dz = scale * np.conj(diff)
            dL_dzbar = scale * diff
            grad_alpha = 0.0
            grad_beta = 0.0
        else:
            y_hat = alpha * z.real + beta * z.imag
            diff = y_hat - y
            loss = float(np.mean(diff ** 2))
            dL_dyhat = 2.0 * diff / len(y)
            dL_dz = dL_dyhat * (alpha / 2 - 0.5j * beta)
            dL_dzbar = dL_dyhat * (alpha / 2 + 0.5j * beta)
            grad_alpha = float(np.mean(dL_dyhat * z.real))
            grad_beta = float(np.mean(dL_dyhat * z.imag))

        grads_d: List[float] = []
        for slot in self._learnable_slots:
            wg = self.grad_fns[slot]
            dz = np.asarray(wg.dz_fn(*d_args), dtype=np.complex128)
            dzb = np.asarray(wg.dz_bar_fn(*d_args), dtype=np.complex128)
            chain = dL_dz * dz + dL_dzbar * dzb
            grads_d.append(float(np.mean(chain.real)))

        return loss, grads_d, grad_alpha, grad_beta

    def apply_symbolic_grads(
        self,
        grads_d: List[float],
        grad_alpha: float,
        grad_beta: float,
        lr: float,
    ) -> None:
        with torch.no_grad():
            for p, g in zip(self.d_params, grads_d):
                p -= lr * g
            self.alpha -= lr * grad_alpha
            self.beta -= lr * grad_beta


EMLModel = CompiledEMLModel
