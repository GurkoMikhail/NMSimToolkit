import numpy as np
import hepunits as units
from numba import njit
from core.physics.interaction_soa import RNGContext
from core.other.typing_definitions import Float, Charge
from core.physics.g4coherent_arrays import PP0, PP1, PP2, PP3, PP4, PP5, PP6, PP7, PP8

x = units.cm / (units.h_Planck * units.c_light)
f_factor = 0.5 * x * x

@njit(cache=True, inline='always')
def _calculate_w(xx: Float, b: Float, n: Float, numlim: Float) -> Float:
    """Calculates the scaling weight factor w."""
    x = 2.0 * xx * b
    return n * x * (1. - 0.5 * (n - 1.0) * x * (1. - (n - 2.0) * x / 3.)) if x < numlim else 1. - np.exp(-n * np.log(1. + x))


@njit(cache=True, inline='always')
def _generate_coherent_theta_scalar(energy: Float, Z: Charge, rng_ctx: RNGContext) -> Float:
    """
    Scalar, in-place random generation of coherent scattering angle theta.
    """
    xx = f_factor * energy * energy

    n0 = PP6[Z] - 1.
    n1 = PP7[Z] - 1.
    n2 = PP8[Z] - 1.
    b0 = PP3[Z]
    b1 = PP4[Z]
    b2 = PP5[Z]

    numlim = 0.02
    w0 = _calculate_w(xx, b0, n0, numlim)
    w1 = _calculate_w(xx, b1, n1, numlim)
    w2 = _calculate_w(xx, b2, n2, numlim)

    x0 = w0 * PP0[Z] / (b0 * n0)
    x1 = w1 * PP1[Z] / (b1 * n1)
    x2 = w2 * PP2[Z] / (b2 * n2)

    next_double = rng_ctx.next_double
    state_addr = rng_ctx.state_addr

    while True:
        w = w0
        n = n0
        b = b0

        r1 = next_double(state_addr)
        x = r1 * (x0 + x1 + x2)
        if x > x0:
            x -= x0
            if x <= x1:
                w = w1
                n = n1
                b = b1
            else:
                w = w2
                n = n2
                b = b2
        n = 1.0 / n

        r2 = next_double(state_addr)
        y = w * r2
        if y < numlim:
            x = y * n * (1. + 0.5 * (n + 1.) * y * (1. - (n + 2.) * y / 3.))
        else:
            x = np.exp(-n * np.log(1. - y)) - 1.0
        cost = 1. - x / (b * xx)

        r3 = next_double(state_addr)
        if 2 * r3 < 1. + cost * cost or cost > -1.0:
            break

    return np.arccos(cost)
