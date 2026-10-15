"""
Модуль генерации полярного угла когерентного (рэлеевского) рассеяния фотонов
по параметризации форм-факторов Geant4.
"""

import hepunits as units
import numpy as np
from numba import njit

from core.other.typing_definitions import Charge, Float
from core.physics.g4coherent_arrays import PP0, PP1, PP2, PP3, PP4, PP5, PP6, PP7, PP8
from core.physics.interaction_buffers import RNGContext

_INV_HC: Float = Float(units.cm / (units.h_Planck * units.c_light))
f_factor: Float = Float(0.5 * _INV_HC * _INV_HC)


@njit(cache=True, inline='always')
def _generate_coherent_theta_scalar(energy: Float, Z: Charge, rng_ctx: RNGContext) -> Float:
    """
    Скалярная генерация полярного угла когерентного рассеяния theta методом неймановского отбора.
    """
    if energy <= 0.0 or Z <= 0:
        return Float(0.0)

    def _calc_w_true(arg_val: Float, power_n: Float) -> Float:
        if arg_val <= 0.0 or power_n <= 0.0:
            return Float(0.0)
        return Float(
            power_n * arg_val * (1.0 - 0.5 * (power_n - 1.0) * arg_val * (1.0 - (power_n - 2.0) * arg_val / 3.0))
        )

    def _calc_w_false(arg_val: Float, power_n: Float) -> Float:
        if arg_val <= 0.0 or power_n <= 0.0:
            return Float(0.0)
        return Float(1.0 - np.exp(-power_n * np.log(1.0 + arg_val)))

    xx = f_factor * energy * energy
    if xx <= 0.0:
        return Float(0.0)

    n0 = PP6[Z] - 1.0
    n1 = PP7[Z] - 1.0
    n2 = PP8[Z] - 1.0
    b0 = PP3[Z]
    b1 = PP4[Z]
    b2 = PP5[Z]

    numlim = 0.02

    x0_val = 2.0 * xx * b0
    w0 = _calc_w_true(x0_val, n0) if x0_val < numlim else _calc_w_false(x0_val, n0)

    x1_val = 2.0 * xx * b1
    w1 = _calc_w_true(x1_val, n1) if x1_val < numlim else _calc_w_false(x1_val, n1)

    x2_val = 2.0 * xx * b2
    w2 = _calc_w_true(x2_val, n2) if x2_val < numlim else _calc_w_false(x2_val, n2)

    denom0 = b0 * n0
    denom1 = b1 * n1
    denom2 = b2 * n2

    x0 = w0 * PP0[Z] / denom0 if denom0 > 0.0 else 0.0
    x1 = w1 * PP1[Z] / denom1 if denom1 > 0.0 else 0.0
    x2 = w2 * PP2[Z] / denom2 if denom2 > 0.0 else 0.0

    total_weight = x0 + x1 + x2
    if total_weight <= 0.0:
        return Float(0.0)

    next_double = rng_ctx.next_double
    state_addr = rng_ctx.state_addr

    while True:
        weight_branch = w0
        power_n = n0
        beta_param = b0

        r1 = next_double(state_addr)
        sample_x = r1 * total_weight
        if sample_x > x0:
            sample_x -= x0
            if sample_x <= x1:
                weight_branch = w1
                power_n = n1
                beta_param = b1
            else:
                weight_branch = w2
                power_n = n2
                beta_param = b2
        inv_n = 1.0 / power_n

        r2 = next_double(state_addr)
        y_val = weight_branch * r2
        if y_val < numlim:
            sample_x = y_val * inv_n * (1.0 + 0.5 * (inv_n + 1.0) * y_val * (1.0 - (inv_n + 2.0) * y_val / 3.0))
        else:
            sample_x = np.exp(-inv_n * np.log(max(1.0 - y_val, 1e-300))) - 1.0

        denom_cost = beta_param * xx
        cost = 1.0 - sample_x / denom_cost if denom_cost > 0.0 else 1.0

        r3 = next_double(state_addr)
        if cost >= -1.0 and 2.0 * r3 < 1.0 + cost * cost:
            break

    cost = min(1.0, max(-1.0, cost))
    return Float(np.arccos(cost))


__all__ = [
    "f_factor",
    "_generate_coherent_theta_scalar",
]

