import numpy as np
import hepunits as units
from numba import njit
import core.physics.g4compton as g4compton
from core.other.typing_definitions import Float, Charge
from core.physics.interaction_soa import RNGContext

# Extract the constants from the old g4compton module
from core.physics.scat_func_fit_param_array import scat_func_fit_param

ln10 = np.log(10.)
electron_mass_c2 = 0.510998910 * units.MeV

@njit(cache=True, inline='always')
def _compute_scattering_function_scalar(x: Float, Z: Charge) -> Float:
    """
    Scalar version of the scattering function computation.
    """
    value = float(Z)
    if x <= scat_func_fit_param[Z][3]:
        lgq = np.log(x) / ln10
        if lgq < scat_func_fit_param[Z][1]:
            value = scat_func_fit_param[Z][4] + lgq * scat_func_fit_param[Z][5]
        elif lgq >= scat_func_fit_param[Z][1] and lgq < scat_func_fit_param[Z][2]:
            value = scat_func_fit_param[Z][6] + \
                lgq * (scat_func_fit_param[Z][7] + \
                    lgq * (scat_func_fit_param[Z][8] + \
                        lgq * (scat_func_fit_param[Z][9] + \
                            lgq * scat_func_fit_param[Z][10])))
        else:
            value = scat_func_fit_param[Z][11] + \
                lgq * (scat_func_fit_param[Z][12] + \
                    lgq * (scat_func_fit_param[Z][13] + \
                        lgq * (scat_func_fit_param[Z][14] + \
                            lgq * scat_func_fit_param[Z][15])))
        value = np.exp(value * ln10)
    return value

@njit(cache=True, inline='always')
def _generate_compton_theta_scalar(energy: Float, Z: Charge, rng_ctx: RNGContext) -> Float:
    """
    Scalar, in-place random generation of Compton scattering angle theta.
    """
    e0m = energy / electron_mass_c2

    epsilon0_local = 1. / (1. + 2. * e0m)
    epsilon0_sq = epsilon0_local * epsilon0_local
    alpha1 = -np.log(epsilon0_local)
    alpha2 = 0.5 * (1. - epsilon0_sq)

    wl_photon = units.h_Planck * units.c_light / energy

    next_double = rng_ctx.next_double
    state_addr = rng_ctx.state_addr

    while True:
        r1 = next_double(state_addr)

        if alpha1 / (alpha1 + alpha2) > r1:
            r2 = next_double(state_addr)
            epsilon = np.exp(-alpha1 * r2)
            epsilon_sq = epsilon * epsilon
        else:
            r3 = next_double(state_addr)
            epsilon_sq = epsilon0_sq + (1. - epsilon0_sq) * r3
            epsilon = np.sqrt(epsilon_sq)

        one_cos_t = (1. - epsilon) / (epsilon * e0m)
        sinT2 = one_cos_t * (2. - one_cos_t)
        x = np.sqrt(one_cos_t / 2.) * units.cm / wl_photon
        scattering_function = _compute_scattering_function_scalar(x, Z)
        g_reject = (1. - epsilon * sinT2 / (1. + epsilon_sq)) * scattering_function

        r4 = next_double(state_addr)
        if g_reject > r4 * Z:
            break

    cos_theta = 1. - one_cos_t
    return np.arccos(cos_theta)

@njit(cache=True, inline='always')
def _calculate_compton_energy_deposit_scalar(theta: Float, particle_energy: Float) -> Float:
    """
    Calculate energy deposit for a given theta.
    """
    k = particle_energy / electron_mass_c2
    k1_cos = k * (1.0 - np.cos(theta))
    energy_deposit = particle_energy * k1_cos / (1.0 + k1_cos)
    return energy_deposit
