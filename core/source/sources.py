from typing import Any, cast, List, Optional, Sequence, Tuple, Union

import numpy as np
import hepunits as units
from numpy.typing import NDArray

import core.other.utils as utils
from core.other.typing_definitions import (Activity, Energy, Float, Length,
                                           Time, Vector3D)
from core.particles.particles import ParticleArray


class Source:
    """
    Класс источника частиц

    [activity] = Bq

    [distribution] = Float[:,:,:]

    [voxel_size] = units.cm

    [energy] = units.eV

    [half_life] = sec
    """

    distribution: NDArray[Float]
    initial_activity: NDArray[Float]
    voxel_size: Length
    size: Vector3D
    radiation_type: str
    energy: np.ndarray
    half_life: Time
    timer: Time
    transformation_matrix: NDArray[Float]
    rng: np.random.Generator
    emission_table: List[NDArray[Any]]

    def __init__(self, distribution: Any, activity: Optional[Any] = None, voxel_size: Length = Float(4 * units.mm), radiation_type: str = 'Gamma', energy: Union[Float, List[List[float]]] = Float(140.5 * units.keV), half_life: Time = Float(6 * units.hour), rng: Optional[np.random.Generator] = None) -> None:
        self.distribution = np.asarray(distribution, dtype=Float)
        self.distribution /= np.sum(self.distribution)
        self.initial_activity = np.sum(distribution) if activity is None else np.asarray(activity, dtype=Float)
        self.voxel_size = voxel_size
        self.size = np.asarray(self.distribution.shape)*self.voxel_size
        self.radiation_type = radiation_type
        
        energy = [[energy, 1.0], ] if not isinstance(energy, list) else energy
        energy_arr = np.array(energy)
        self.energy = np.zeros(energy_arr.shape[0], dtype=[("energy", Float), ("probability", Float)])
        self.energy["energy"] = cast(NDArray[Float], energy_arr[:, 0])
        self.energy["probability"] = energy_arr[:, 1]
        self.energy["probability"] /= np.sum(self.energy["probability"])
        
        self.half_life = half_life
        self.timer = Float(0.)
        self._generate_emission_table()
        self.transformation_matrix = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
        self.rng = np.random.default_rng() if rng is None else rng

    def translate(self, x: Float = Float(0.), y: Float = Float(0.), z: Float = Float(0.), in_local: bool = False) -> None:
        """ Переместить объём """
        translation = np.asarray([x, y, z])
        translation_matrix = utils.compute_translation_matrix(translation)
        if in_local:
            self.transformation_matrix = self.transformation_matrix @ translation_matrix
        else:
            self.transformation_matrix = translation_matrix @ self.transformation_matrix

    def rotate(self, alpha: Float = Float(0.), beta: Float = Float(0.), gamma: Float = Float(0.), rotation_center: Sequence[Float] = (Float(0.), Float(0.), Float(0.)), in_local: bool = False) -> None:
        """ Повернуть объём вокруг координатных осей """
        rotation_angles = np.asarray([alpha, beta, gamma])
        rotation_center_arr = np.asarray(rotation_center)
        rotation_matrix = utils.compute_translation_matrix(rotation_center_arr)
        rotation_matrix = rotation_matrix @ utils.compute_rotation_matrix(rotation_angles)
        rotation_matrix = rotation_matrix @ utils.compute_translation_matrix(-rotation_center_arr)
        if in_local:
            self.transformation_matrix = self.transformation_matrix @ rotation_matrix
        else:
            self.transformation_matrix = rotation_matrix @ self.transformation_matrix

    def convert_to_global_position(self, position: NDArray[Float]) -> NDArray[Float]:
        global_position = np.ones((position.shape[0], 4), dtype=position.dtype)
        global_position[:, :3] = position
        np.matmul(global_position, self.transformation_matrix.T.astype(position.dtype), out=global_position)
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
    def activity(self) -> NDArray[Float]:
        return self.initial_activity * 2 ** (-self.timer / self.half_life)
    
    @property
    def nuclei_number(self) -> NDArray[Float]:
        return self.activity * self.half_life / np.log(2)

    def set_state(self, timer: Optional[Time], rng_state: Optional[Any] = None) -> None:
        if timer is not None:
            self.timer = timer
        if rng_state is None:
            return
        self.rng.bit_generator.state['state'] = rng_state # type: ignore

    def generate_energy(self, n: int) -> NDArray[Float]:
        energy = self.rng.choice(self.energy["energy"], n, p=self.energy["probability"])
        return energy

    def generate_position(self, n: int) -> Vector3D:
        position = self.emission_table[0]
        probability = self.emission_table[1]
        position = self.rng.choice(position, n, p=probability)
        position += self.rng.uniform(0., self.voxel_size, position.shape)
        position = self.convert_to_global_position(position)
        return position

    def generate_emission_time(self, n: int) -> Tuple[NDArray[Float], Float]:
        dt = np.log((self.nuclei_number + n) / self.nuclei_number) * self.half_life / np.log(2)
        a = 2 ** (-self.timer / self.half_life)
        b = 2 ** (-(self.timer + dt) / self.half_life)
        alpha = self.rng.uniform(b, a, n)
        emission_time = -np.log(alpha) * self.half_life / np.log(2)
        return emission_time, Float(dt)

    def generate_direction(self, n: int) -> Vector3D:
        a1 = self.rng.random(n)
        a2 = self.rng.random(n)
        cos_alpha = 1 - 2 * a1
        sq = np.sqrt(1 - cos_alpha ** 2)
        cos_beta = sq * np.cos(2 * np.pi * a2)
        cos_gamma = sq * np.sin(2 * np.pi * a2)
        direction = np.column_stack((cos_alpha, cos_beta, cos_gamma))
        return direction

    def generate_particles(self, n: int) -> ParticleArray:
        energy = self.generate_energy(n)
        direction = self.generate_direction(n)
        position = self.generate_position(n)
        emission_time, dt = self.generate_emission_time(n)
        self.timer += dt
        particles = ParticleArray(np.zeros_like(energy, dtype=np.uint64), position, direction, energy, emission_time)
        return particles


class PointSource(Source):
    """
    Точечный источник

    [position = (x, y, z)] = units.cm

    [activity] = Bq
    
    [energy] = units.eV
    """

    def __init__(self, activity, energy, size=1.*units.mm, half_life=6.*units.hour, rng=None):
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

    [position = (x, y, z)] = units.cm

    [activity] = Bq

    [distribution] = Float[:,:,:]

    [voxel_size] = units.cm
    """

    def __init__(self, distribution, activity=None, voxel_size=4*units.mm):
        radiation_type = 'Gamma'
        energy = Float(140.5 * units.keV)
        half_life = 6.*units.hour
        super().__init__(distribution, activity, voxel_size, radiation_type, energy, half_life)

class I123(Source):
    """
    Источник I123

    [position = (x, y, z)] = units.cm

    [activity] = Bq

    [distribution] = Float[:,:,:]

    [voxel_size] = units.cm
    """

    def __init__(self, distribution, activity=None, voxel_size=4*units.mm):
        radiation_type = 'Gamma'
        energy = [
            [158.97*units.keV, 83.0],
            [528.96*units.keV, 1.39],
            [440.02*units.keV, 0.428],
            [538.54*units.keV, 0.382],
            [505.33*units.keV, 0.316],
            [346.35*units.keV, 0.126],
        ]
        half_life = 13.27*units.hour
        super().__init__(distribution, activity, voxel_size, radiation_type, energy, half_life)


class SourcePhantom(Тс99m_MIBI):
    """
    Источник 99mТс-MIBI

    [position = (x, y, z)] = units.cm

    [activity] = Bq

    [phantom_name] = string

    [voxel_size] = units.cm
    """

    def __init__(self, phantom_name: str, activity: Optional[Float] = None, voxel_size: Float = Float(4 * units.mm)) -> None:
        distribution = np.load(f'Phantoms/{phantom_name}.npy')
        super().__init__(distribution, activity, voxel_size)


class efg3(SourcePhantom):
    """
    Источник efg3

    [position = (x, y, z)] = units.cm

    [activity] = Bq
    """

    def __init__(self, activity):
        phantom_name = 'efg3'
        voxel_size = 4.*units.mm
        super().__init__(phantom_name, activity, voxel_size)
        

class efg3cut(SourcePhantom):
    """
    Источник efg3cut

    [position = (x, y, z)] = units.cm

    [activity] = Bq
    """

    def __init__(self, activity):
        phantom_name = 'efg3cut'
        voxel_size = 4.*units.mm
        super().__init__(phantom_name, activity, voxel_size)


class efg3cutDefect(SourcePhantom):
    """
    Источник efg3cutDefect

    [position = (x, y, z)] = units.cm

    [activity] = Bq
    """

    def __init__(self, position, activity, rotation_angles=None, rotation_center=None):
        phantom_name = 'efg3cutDefect'
        voxel_size = 4.*units.mm
        super().__init__(position, activity, phantom_name, voxel_size, rotation_angles, rotation_center)
