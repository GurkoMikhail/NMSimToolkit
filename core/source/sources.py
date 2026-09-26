from typing import Any, cast, List, Optional, Sequence, Tuple, Union

import numpy as np
import hepunits as units
from numpy.typing import NDArray

import core.other.utils as utils
from core.other.typing_definitions import Float, Length, Time, Species, Index
from core.other.vectors import Vector3DSoA
from core.particles.particles import ParticleBank
from core.scene.nodes import CompositeNode


class Source(CompositeNode):
    """
    Класс источника частиц для Data-Oriented Design (SoA)

    [activity] = Bq

    [distribution] = Float[:,:,:]

    [voxel_size] = units.cm

    [energy] = units.eV

    [half_life] = sec
    """

    _distribution: NDArray[Float]
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

    def __init__(self, distribution: Any, activity: Optional[Any] = None, voxel_size: Length = Float(4 * units.mm), radiation_type: str = 'Gamma', energy: Union[Float, List[List[Float]]] = Float(140.5 * units.keV), half_life: Time = Float(6 * units.hour), rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        dist_array = np.asarray(distribution, dtype=Float)
        total_activity = Float(np.sum(dist_array))
        self.initial_activity = total_activity if activity is None else np.asarray(activity, dtype=Float)
        self._voxel_size = Float(voxel_size)
        self.radiation_type = radiation_type
        self.size = np.zeros(3, dtype=Float)
        self.emission_table = []
        self._distribution = np.empty((0, 0, 0), dtype=Float)
        self.distribution = dist_array

        energy = [[energy, Float(1.0)], ] if not isinstance(energy, list) else energy
        energy_arr = np.array(energy)
        self.energy = np.zeros(energy_arr.shape[0], dtype=[("energy", Float), ("probability", Float)])
        self.energy["energy"] = cast(NDArray[Float], energy_arr[:, 0])
        self.energy["probability"] = energy_arr[:, 1]
        en_prob_sum = np.sum(self.energy["probability"])
        if en_prob_sum > 0:
            self.energy["probability"] /= en_prob_sum

        self.half_life = half_life
        self.rng = np.random.default_rng() if rng is None else rng

    @property
    def distribution(self) -> NDArray[Float]:
        """Нормированное пространственное распределение вероятностей испускания частиц."""
        return self._distribution

    @distribution.setter
    def distribution(self, value: Any) -> None:
        dist_array = np.asarray(value, dtype=Float)
        if dist_array.size == 0:
            self._distribution = dist_array
            self.size = np.zeros(3, dtype=Float)
            self._generate_emission_table()
            return
        total_activity = Float(np.sum(dist_array))
        if total_activity <= 0.0:
            raise ValueError("Распределение активности не может быть нулевым или отрицательным — источник не содержит активного вещества.")
        self._distribution = dist_array / total_activity
        self.size = np.asarray(self._distribution.shape) * self._voxel_size
        self._generate_emission_table()

    @property
    def voxel_size(self) -> Length:
        """Шаг воксельной сетки распределения активности (мм)."""
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, value: Length) -> None:
        self._voxel_size = Float(value)
        if self._distribution is not None and self._distribution.size > 0:
            self.size = np.asarray(self._distribution.shape) * self._voxel_size
            self._generate_emission_table()

    def _generate_emission_table(self) -> None:
        """
        Генерация таблицы координат и нормированных вероятностей испускания для непустых вокселей.
        Гарантирует, что сумма вероятностей строго равна 1.0 (без машинной погрешности округления).
        """
        if self._distribution.size == 0:
            self.emission_table = [np.zeros((0, 3), dtype=Float), np.zeros(0, dtype=Float)]
            return

        grid_x, grid_y, grid_z = np.meshgrid(
            np.linspace(0, self.size[0], self._distribution.shape[0], endpoint=False),
            np.linspace(0, self.size[1], self._distribution.shape[1], endpoint=False),
            np.linspace(0, self.size[2], self._distribution.shape[2], endpoint=False),
            indexing='ij'
        )
        position = np.stack((grid_x, grid_y, grid_z), axis=3).reshape(-1, 3) - self.size / 2
        probability = self._distribution.ravel()
        indices = probability.nonzero()[0]
        prob_values = probability[indices]
        prob_sum = np.sum(prob_values)
        if prob_sum > 0:
            prob_values = prob_values / prob_sum
        self.emission_table = [position[indices], prob_values]

    @property
    def decay_constant(self) -> Float:
        if np.isinf(self.half_life):
            return Float(0.0)
        return Float(np.log(2) / self.half_life)

    def _get_effective_dt(self, dt: Float) -> Float:
        lambd = self.decay_constant
        if lambd == 0.0 or lambd * dt < 1e-6:
            return dt
        return Float((1.0 - np.exp(-lambd * dt)) / lambd)

    def get_activity(self, t: Float) -> Float:
        """Мгновенная активность источника в момент времени t."""
        if self.decay_constant == 0.0:
            return Float(self.initial_activity)
        return Float(self.initial_activity * (2.0 ** (-t / self.half_life)))

    def get_expected_particles(self, t1: Float, t2: Float) -> Float:
        """Точный интеграл распада на интервале [t1, t2]."""
        dt = t2 - t1
        if dt <= 0:
            return Float(0.0)
        return Float(self.get_activity(t1) * self._get_effective_dt(dt))

    def set_state(self, rng_state: Optional[Any] = None) -> None:
        if rng_state is None:
            return
        self.rng.bit_generator.state['state'] = rng_state# type: ignore

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

    def generate_emission_time(self, n: int, t1: Float, t2: Float) -> NDArray[Float]:
        dt = Float(t2 - t1)
        u = self.rng.uniform(0.0, 1.0, n)
        lambd = self.decay_constant

        if lambd == 0.0 or lambd * dt < 1e-6:
            # Linear approximation for stable sources
            emission_time = t1 + u * dt
        else:
            # Exact inverse CDF
            effective_dt = self._get_effective_dt(dt)
            emission_time = t1 - (1.0 / lambd) * np.log(1.0 - u * lambd * effective_dt)
        return emission_time

    def generate_direction(self, n: int) -> Vector3D:
        a1 = self.rng.random(n)
        a2 = self.rng.random(n)
        cos_alpha = 1 - 2 * a1
        sq = np.sqrt(1 - cos_alpha ** 2)
        cos_beta = sq * np.cos(2 * np.pi * a2)
        cos_gamma = sq * np.sin(2 * np.pi * a2)
        direction = np.column_stack((cos_alpha, cos_beta, cos_gamma))
        return direction

    def inject(self, bank: ParticleBank, batch_size: int, t1: Float, t2: Float) -> NDArray[Index]:
        """
        Генерирует частицы и инжектирует их напрямую в банк через механизм Direct Injection (SoA).
        """
        n = min(batch_size, bank.capacity - len(bank.active_indices))
        if n <= 0:
            return np.array([], dtype=Index)

        energy = self.generate_energy(n)
        direction_arr = self.generate_direction(n)
        position_arr = self.generate_position(n)
        emission_time = self.generate_emission_time(n, t1, t2)

        position = Vector3DSoA(
            x=position_arr[:, 0].astype(Length),
            y=position_arr[:, 1].astype(Length),
            z=position_arr[:, 2].astype(Length)
        )
        direction = Vector3DSoA(
            x=direction_arr[:, 0].astype(Float),
            y=direction_arr[:, 1].astype(Float),
            z=direction_arr[:, 2].astype(Float)
        )

        species = np.zeros(n, dtype=Species)
        distance_traveled = np.zeros(n, dtype=Length)

        target_indices = bank.inject_particles(
            species=species,
            position=position,
            direction=direction,
            energy=energy,
            emission_time=emission_time,
            distance_traveled=distance_traveled
        )
        return target_indices


class PointSource(Source):
    """
    Точечный источник (SoA)

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


class Tc99m_MIBI(Source):
    """
    Источник 99mTc-MIBI (SoA)

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
    Источник I123 (SoA)

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


class SourcePhantom(Tc99m_MIBI):
    """
    Источник 99mTc-MIBI (SoA)

    [position = (x, y, z)] = units.cm

    [activity] = Bq

    [phantom_name] = string

    [voxel_size] = units.cm
    """

    def __init__(self, phantom_name: str, activity: Optional[Float] = None, voxel_size: Float = Float(4 * units.mm)) -> None:
        distribution = np.load(f'Phantoms/{phantom_name}.npy', allow_pickle=True)
        super().__init__(distribution, activity, voxel_size)


class efg3(SourcePhantom):
    """
    Источник efg3 (SoA)

    [position = (x, y, z)] = units.cm

    [activity] = Bq
    """

    def __init__(self, activity):
        phantom_name = 'efg3'
        voxel_size = 4.*units.mm
        super().__init__(phantom_name, activity, voxel_size)


class efg3cut(SourcePhantom):
    """
    Источник efg3cut (SoA)

    [position = (x, y, z)] = units.cm

    [activity] = Bq
    """

    def __init__(self, activity):
        phantom_name = 'efg3cut'
        voxel_size = 4.*units.mm
        super().__init__(phantom_name, activity, voxel_size)


class efg3cutDefect(SourcePhantom):
    """
    Источник efg3cutDefect (SoA)

    [position = (x, y, z)] = units.cm

    [activity] = Bq
    """

    def __init__(self, position, activity, rotation_angles=None, rotation_center=None):
        phantom_name = 'efg3cutDefect'
        voxel_size = 4.*units.mm
        super().__init__(phantom_name, activity, voxel_size)

        # Apply optional position and rotation transformations after initialization
        if rotation_angles is not None:
            if rotation_center is None:
                rotation_center = (0.0, 0.0, 0.0)
            self.rotate(rotation_angles[0], rotation_angles[1], rotation_angles[2], rotation_center=rotation_center)
        if position is not None:
            self.translate(position[0], position[1], position[2])
