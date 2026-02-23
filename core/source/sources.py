import numpy as np
from core.particles.particles import ParticleArray
from numpy import load, cos, sin, log, sqrt, matmul
from typing import List, Optional, Any, Union, Tuple, Sequence
from numpy.typing import NDArray
import core.other.utils as utils
from hepunits import*
from core.other.typing_definitions import Length, Activity, Energy, Time, Vector3D, Float


class Source:
    """
    Класс источника частиц

    [activity] = Bq

    [distribution] = float[:,:,:]

    [voxel_size] = cm

    [energy] = eV

    [half_life] = sec
    """

    distribution: NDArray[np.float64]
    initial_activity: NDArray[np.float64]
    voxel_size: Length
    size: Vector3D # type: ignore
    radiation_type: str
    energy: np.ndarray
    half_life: Time
    timer: Time
    transformation_matrix: NDArray[np.float64]
    rng: np.random.Generator
    emission_table: List[NDArray[Any]]

    def __init__(self, distribution: Any, activity: Optional[Any] = None, voxel_size: Length = 4*mm, radiation_type: str = 'Gamma', energy: Union[float, List[List[float]]] = 140.5*keV, half_life: Time = 6*hour, rng: Optional[np.random.Generator] = None, precision: Any = Float) -> None:
        self.distribution = np.asarray(distribution, dtype=precision)
        self.distribution /= np.sum(self.distribution)
        self.initial_activity = np.sum(distribution) if activity is None else np.asarray(activity, dtype=precision)
        self.voxel_size = voxel_size
        self.size = np.asarray(self.distribution.shape)*self.voxel_size
        self.radiation_type = radiation_type
        
        energy = [[energy, 1.0], ] if not isinstance(energy, list) else energy
        energy_arr = np.array(energy)
        self.energy = np.zeros(energy_arr.shape[0], dtype=[("energy", "d"), ("probability", "d")])
        self.energy["energy"] = energy_arr[:, 0]
        self.energy["probability"] = energy_arr[:, 1]
        self.energy["probability"] /= np.sum(self.energy["probability"])
        
        self.half_life = half_life
        self.timer = 0.
        self._generate_emission_table()
        self.transformation_matrix = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
        self.rng = np.random.default_rng() if rng is None else rng

    def translate(self, x=0., y=0., z=0., in_local=False):
        """ Переместить объём """
        translation = np.asarray([x, y, z])
        translation_matrix = utils.compute_translation_matrix(translation)
        if in_local:
            self.transformation_matrix = self.transformation_matrix@translation_matrix
        else:
            self.transformation_matrix = translation_matrix@self.transformation_matrix

    def rotate(self, alpha=0., beta=0., gamma=0., rotation_center=[0., 0., 0.], in_local=False):
        """ Повернуть объём вокруг координатных осей """
        rotation_angles = np.asarray([alpha, beta, gamma])
        rotation_center_arr = np.asarray(rotation_center)
        rotation_matrix = utils.compute_translation_matrix(rotation_center_arr)
        rotation_matrix = rotation_matrix@utils.compute_rotation_matrix(rotation_angles)
        rotation_matrix = rotation_matrix@utils.compute_translation_matrix(-rotation_center_arr)
        if in_local:
            self.transformation_matrix = self.transformation_matrix@rotation_matrix
        else:
            self.transformation_matrix = rotation_matrix@self.transformation_matrix

    def convert_to_global_position(self, position):
        global_position = np.ones((position.shape[0], 4), dtype=position.dtype)
        global_position[:, :3] = position
        matmul(global_position, self.transformation_matrix.T.astype(position.dtype), out=global_position)
        return global_position[:, :3]

    def _generate_emission_table(self):
        xs, ys, zs = np.meshgrid(
            np.linspace(0, self.size[0], self.distribution.shape[0], endpoint=False),
            np.linspace(0, self.size[1], self.distribution.shape[1], endpoint=False),
            np.linspace(0, self.size[2], self.distribution.shape[2], endpoint=False),
            indexing = 'ij'
        )
        position = np.stack((xs, ys, zs), axis=3).reshape(-1, 3) - self.size/2
        probability = self.distribution.ravel()
        indices = probability.nonzero()[0]
        self.emission_table = [position[indices], probability[indices]]

    @property
    def activity(self):
        return self.initial_activity*2**(-self.timer/self.half_life)
    
    @property
    def nuclei_number(self):
        return self.activity*self.half_life/log(2)

    def set_state(self, timer, rng_state=None):
        if timer is not None:
            self.timer = timer
        if rng_state is None:
            return
        self.rng.bit_generator.state['state'] = rng_state

    def generate_energy(self, n):
        energy = self.rng.choice(self.energy["energy"], n, p=self.energy["probability"])
        return energy

    def generate_position(self, n):
        position = self.emission_table[0]
        probability = self.emission_table[1]
        position = self.rng.choice(position, n, p=probability)
        position += self.rng.uniform(0., self.voxel_size, position.shape)
        position = self.convert_to_global_position(position)
        return position

    def generate_emission_time(self, n):
        dt = log((self.nuclei_number + n)/self.nuclei_number)*self.half_life/log(2)
        a = 2**(-self.timer/self.half_life)
        b = 2**(-(self.timer + dt)/self.half_life)
        alpha = self.rng.uniform(b, a, n)
        emission_time = -log(alpha)*self.half_life/log(2)
        return emission_time, dt

    def generate_direction(self, n):
        a1 = self.rng.random(n)
        a2 = self.rng.random(n)
        cos_alpha = 1 - 2*a1
        sq = sqrt(1 - cos_alpha**2)
        cos_beta = sq*cos(2*pi*a2)
        cos_gamma = sq*sin(2*pi*a2)
        direction = np.column_stack((cos_alpha, cos_beta, cos_gamma))
        return direction

    def generate_particles(self, n: int) -> ParticleArray: # type: ignore
        energy = self.generate_energy(n)
        direction = self.generate_direction(n)
        position = self.generate_position(n)
        emission_time, dt = self.generate_emission_time(n)
        self.timer += dt
        particles = ParticleArray(np.zeros_like(energy), position, direction, energy, emission_time)
        return particles


class PointSource(Source):
    """
    Точечный источник

    [position = (x, y, z)] = cm

    [activity] = Bq
    
    [energy] = eV
    """

    def __init__(self, activity, energy, size=1.*mm, half_life=6.*hour, rng=None):
        distribution = [[[1.]]]
        super().__init__(
            distribution=distribution,
            activity=activity,
            voxel_size=size,
            energy=energy,
            half_life=half_life,
            rng=rng
        )

class Тс99m_MIBI(Source):
    """
    Источник 99mТс-MIBI

    [position = (x, y, z)] = cm

    [activity] = Bq

    [distribution] = float[:,:,:]

    [voxel_size] = cm
    """

    def __init__(self, distribution, activity=None, voxel_size=4*mm):
        radiation_type = 'Gamma'
        energy = 140.5*keV
        half_life = 6.*hour
        super().__init__(distribution, activity, voxel_size, radiation_type, energy, half_life)

class I123(Source):
    """
    Источник I123

    [position = (x, y, z)] = cm

    [activity] = Bq

    [distribution] = float[:,:,:]

    [voxel_size] = cm
    """

    def __init__(self, distribution, activity=None, voxel_size=4*mm):
        radiation_type = 'Gamma'
        energy = [
            [158.97*keV, 83.0],
            [528.96*keV, 1.39],
            [440.02*keV, 0.428],
            [538.54*keV, 0.382],
            [505.33*keV, 0.316],
            [346.35*keV, 0.126],
        ]
        half_life = 13.27*hour
        super().__init__(distribution, activity, voxel_size, radiation_type, energy, half_life)


class SourcePhantom(Тс99m_MIBI):
    """
    Источник 99mТс-MIBI

    [position = (x, y, z)] = cm

    [activity] = Bq

    [phantom_name] = string

    [voxel_size] = cm
    """

    def __init__(self, phantom_name, activity=None, voxel_size=4*mm):
        distribution = load(f'Phantoms/{phantom_name}.npy')
        super().__init__(distribution, activity, voxel_size)


class efg3(SourcePhantom):
    """
    Источник efg3

    [position = (x, y, z)] = cm

    [activity] = Bq
    """

    def __init__(self, activity):
        phantom_name = 'efg3'
        voxel_size = 4.*mm
        super().__init__(phantom_name, activity, voxel_size)
        

class efg3cut(SourcePhantom):
    """
    Источник efg3cut

    [position = (x, y, z)] = cm

    [activity] = Bq
    """

    def __init__(self, activity):
        phantom_name = 'efg3cut'
        voxel_size = 4.*mm
        super().__init__(phantom_name, activity, voxel_size)


class efg3cutDefect(SourcePhantom):
    """
    Источник efg3cutDefect

    [position = (x, y, z)] = cm

    [activity] = Bq
    """

    def __init__(self, position, activity, rotation_angles=None, rotation_center=None):
        phantom_name = 'efg3cutDefect'
        voxel_size = 4.*mm
        super().__init__(position, activity, phantom_name, voxel_size, rotation_angles, rotation_center)
