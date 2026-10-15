"""
Тесты источников излучения (PointSource, Source) для архитектуры SoA.
"""

import os
import sys
import unittest

import hepunits as units
import numpy as np

from core.other.typing_definitions import Energy, Float, Index, Length, Species, Time
from core.particles.particles import ParticleBank
from core.source.sources import PointSource, Source


class TestSourcesSoA(unittest.TestCase):
    """Тестирование генерации и инжекции частиц из источников излучения."""

    def setUp(self) -> None:
        self.capacity = 10
        self.bank = ParticleBank.allocate(self.capacity)
        self.rng = np.random.default_rng(42)
        self.source = PointSource(activity=Float(1000.0), energy=Float(140.5 * 1000.0), rng=self.rng)

    def test_inject_basic(self) -> None:
        """Проверка базовой инжекции пакета частиц в ParticleBank."""
        batch_size = 5
        time_start = Float(0.0)
        time_end = Float(1.0)
        indices = self.source.inject(self.bank, batch_size, time_start, time_end)

        self.assertEqual(len(indices), batch_size)
        self.assertEqual(self.bank.count, batch_size)
        self.assertEqual(len(self.bank.active_indices), batch_size)

        active_mask = self.bank.state.is_active

        self.assertEqual(self.bank.state.species[active_mask].dtype, Species)
        np.testing.assert_array_equal(
            self.bank.state.species[active_mask], np.zeros(batch_size, dtype=Species)
        )

        self.assertEqual(self.bank.state.distance_traveled[active_mask].dtype, Length)
        np.testing.assert_array_equal(
            self.bank.state.distance_traveled[active_mask], np.zeros(batch_size, dtype=Length)
        )

        np.testing.assert_array_equal(
            self.bank.state.position.x[active_mask],
            self.bank.initial_state.emission_position.x[active_mask],
        )
        np.testing.assert_array_equal(
            self.bank.state.position.y[active_mask],
            self.bank.initial_state.emission_position.y[active_mask],
        )
        np.testing.assert_array_equal(
            self.bank.state.position.z[active_mask],
            self.bank.initial_state.emission_position.z[active_mask],
        )

        np.testing.assert_array_equal(
            self.bank.state.direction.x[active_mask],
            self.bank.initial_state.emission_direction.x[active_mask],
        )
        np.testing.assert_array_equal(
            self.bank.state.direction.y[active_mask],
            self.bank.initial_state.emission_direction.y[active_mask],
        )
        np.testing.assert_array_equal(
            self.bank.state.direction.z[active_mask],
            self.bank.initial_state.emission_direction.z[active_mask],
        )

        self.assertTrue(np.all(self.bank.initial_state.emission_time[active_mask] >= time_start))
        self.assertTrue(np.all(self.bank.initial_state.emission_time[active_mask] <= time_end))

    def test_inject_exceeds_capacity(self) -> None:
        """Проверка ограничения инжекции остаточной емкостью ParticleBank."""
        self.source.inject(self.bank, 7, Float(0.0), Float(1.0))
        self.assertEqual(self.bank.count, 7)
        self.assertEqual(len(self.bank.active_indices), 7)

        batch_size = 5
        indices = self.source.inject(self.bank, batch_size, Float(1.0), Float(2.0))

        self.assertEqual(len(indices), 3)
        self.assertEqual(self.bank.count, 10)
        self.assertEqual(len(self.bank.active_indices), 10)

    def test_inject_no_capacity(self) -> None:
        """Проверка инжекции при полностью заполненном банке частиц."""
        self.source.inject(self.bank, self.capacity, Float(0.0), Float(1.0))
        self.assertEqual(self.bank.count, self.capacity)

        batch_size = 5
        indices = self.source.inject(self.bank, batch_size, Float(1.0), Float(2.0))

        self.assertEqual(len(indices), 0)
        self.assertEqual(self.bank.count, self.capacity)

    def test_voxel_source_sampling(self) -> None:
        """Проверка воспроизводимой выборки координат воксельного источника с изолированным ГСЧ."""
        rng_first = np.random.default_rng(42)
        rng_second = np.random.default_rng(42)
        dist = np.zeros((4, 4, 4), dtype=Float)
        dist[3, 3, 3] = 1.0  # Активен только один угловой воксель [2.0, 4.0]^3 при voxel_size=2.0

        src_first = Source(distribution=dist, voxel_size=Float(2.0 * units.mm), rng=rng_first)
        src_second = Source(distribution=dist, voxel_size=Float(2.0 * units.mm), rng=rng_second)

        pos_first = src_first.generate_position(100)
        pos_second = src_second.generate_position(100)

        np.testing.assert_allclose(pos_first, pos_second)
        self.assertTrue(np.all(pos_first >= 2.0))
        self.assertTrue(np.all(pos_first <= 4.0))

    def test_source_unnormalized_distribution_and_activity(self) -> None:
        """Проверка автоматической нормировки распределения и независимости общей активности."""
        raw_dist = np.array(
            [
                [[1000.0, 2000.0], [3000.0, 4000.0]],
                [[5000.0, 6000.0], [7000.0, 22000.0]],
            ],
            dtype=float,
        )
        self.assertEqual(np.sum(raw_dist), 50000.0)

        rng = np.random.default_rng(42)
        src = Source(distribution=raw_dist, activity=Float(10 * units.MBq), rng=rng)
        self.assertAlmostEqual(float(np.sum(src.distribution)), 1.0, places=7)
        self.assertAlmostEqual(float(np.sum(src.emission_table[1])), 1.0, places=7)
        self.assertAlmostEqual(float(src.initial_activity), float(10 * units.MBq))

        pos = src.generate_position(50)
        self.assertEqual(pos.shape, (50, 3))

        new_dist = np.ones((8, 8, 8)) * 42.0
        src.distribution = new_dist
        self.assertAlmostEqual(float(np.sum(src.distribution)), 1.0, places=7)
        self.assertAlmostEqual(float(np.sum(src.emission_table[1])), 1.0, places=7)
        pos2 = src.generate_position(20)
        self.assertEqual(pos2.shape, (20, 3))

        src2 = Source(distribution=raw_dist, rng=np.random.default_rng(42))
        self.assertAlmostEqual(float(src2.initial_activity), 50000.0)
        self.assertAlmostEqual(float(np.sum(src2.distribution)), 1.0, places=7)


if __name__ == "__main__":
    unittest.main()

