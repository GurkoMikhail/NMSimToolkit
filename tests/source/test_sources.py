import unittest
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.source.sources import PointSource
from core.particles.particles import ParticleBank
from core.other.typing_definitions import Index, Length, Float, Energy, Time, Species


class TestSourcesSoA(unittest.TestCase):
    def setUp(self):
        self.capacity = 10
        self.bank = ParticleBank.allocate(self.capacity)
        # Point source at origin, small size
        self.source = PointSource(activity=1000, energy=140.5 * 1000)

    def test_inject_basic(self):
        # Inject fewer particles than capacity
        batch_size = 5
        t1 = Float(0.0)
        t2 = Float(1.0)
        indices = self.source.inject(self.bank, batch_size, t1, t2)

        self.assertEqual(len(indices), batch_size)
        self.assertEqual(self.bank.count, batch_size)
        self.assertEqual(len(self.bank.active_indices), batch_size)

        # Verify injected values are valid types and values
        active_mask = self.bank.state.is_active

        # Species
        self.assertEqual(self.bank.state.species[active_mask].dtype, Species)
        np.testing.assert_array_equal(self.bank.state.species[active_mask], np.zeros(batch_size, dtype=Species))

        # Distance traveled
        self.assertEqual(self.bank.state.distance_traveled[active_mask].dtype, Length)
        np.testing.assert_array_equal(self.bank.state.distance_traveled[active_mask], np.zeros(batch_size, dtype=Length))

        # Positions and Emission Positions (should be identical)
        np.testing.assert_array_equal(self.bank.state.position.x[active_mask], self.bank.initial_state.emission_position.x[active_mask])
        np.testing.assert_array_equal(self.bank.state.position.y[active_mask], self.bank.initial_state.emission_position.y[active_mask])
        np.testing.assert_array_equal(self.bank.state.position.z[active_mask], self.bank.initial_state.emission_position.z[active_mask])

        # Directions and Emission Directions (should be identical)
        np.testing.assert_array_equal(self.bank.state.direction.x[active_mask], self.bank.initial_state.emission_direction.x[active_mask])
        np.testing.assert_array_equal(self.bank.state.direction.y[active_mask], self.bank.initial_state.emission_direction.y[active_mask])
        np.testing.assert_array_equal(self.bank.state.direction.z[active_mask], self.bank.initial_state.emission_direction.z[active_mask])

        # Emission time should be within [t1, t2]
        self.assertTrue(np.all(self.bank.initial_state.emission_time[active_mask] >= t1))
        self.assertTrue(np.all(self.bank.initial_state.emission_time[active_mask] <= t2))

    def test_inject_exceeds_capacity(self):
        # Fill partially
        self.source.inject(self.bank, 7, Float(0.0), Float(1.0))
        self.assertEqual(self.bank.count, 7)
        self.assertEqual(len(self.bank.active_indices), 7)

        # Try to inject more than remaining capacity
        batch_size = 5
        indices = self.source.inject(self.bank, batch_size, Float(1.0), Float(2.0))

        # Should only inject what fits (10 - 7 = 3)
        self.assertEqual(len(indices), 3)
        self.assertEqual(self.bank.count, 10)
        self.assertEqual(len(self.bank.active_indices), 10)

    def test_inject_no_capacity(self):
        # Fill completely
        self.source.inject(self.bank, self.capacity, Float(0.0), Float(1.0))
        self.assertEqual(self.bank.count, self.capacity)

        # Try to inject more
        batch_size = 5
        indices = self.source.inject(self.bank, batch_size, Float(1.0), Float(2.0))

        self.assertEqual(len(indices), 0)
        self.assertEqual(self.bank.count, self.capacity)

    def test_source_unnormalized_distribution_and_activity(self):
        """Проверка автоматической нормировки распределения и независимости общей активности."""
        from core.source.sources import Source
        import hepunits as units

        # Матрица активности с суммой элементов != 1 (например 50000.0)
        raw_dist = np.array([
            [[1000.0, 2000.0], [3000.0, 4000.0]],
            [[5000.0, 6000.0], [7000.0, 22000.0]]
        ], dtype=float)
        self.assertEqual(np.sum(raw_dist), 50000.0)

        # 1. Задаем источник с явной общей активностью 10 МБк
        src = Source(distribution=raw_dist, activity=Float(10 * units.MBq))
        # Пространственное распределение должно быть отнормировано к 1
        self.assertAlmostEqual(float(np.sum(src.distribution)), 1.0, places=7)
        # Вектор вероятностей в таблице испускания также строго равен 1
        self.assertAlmostEqual(float(np.sum(src.emission_table[1])), 1.0, places=7)
        # Общая активность равна заданным 10 МБк
        self.assertAlmostEqual(float(src.initial_activity), float(10 * units.MBq))

        # 2. Генерация позиций не должна вызывать ошибку 'probabilities do not sum to 1'
        pos = src.generate_position(50)
        self.assertEqual(pos.shape, (50, 3))

        # 3. Присвоение нового распределения с произвольной суммой через сеттер
        new_dist = np.ones((8, 8, 8)) * 42.0
        src.distribution = new_dist
        self.assertAlmostEqual(float(np.sum(src.distribution)), 1.0, places=7)
        self.assertAlmostEqual(float(np.sum(src.emission_table[1])), 1.0, places=7)
        pos2 = src.generate_position(20)
        self.assertEqual(pos2.shape, (20, 3))

        # 4. Если активность не задана, она берется из суммы вокселей до нормировки
        src2 = Source(distribution=raw_dist)
        self.assertAlmostEqual(float(src2.initial_activity), 50000.0)
        self.assertAlmostEqual(float(np.sum(src2.distribution)), 1.0, places=7)

if __name__ == '__main__':
    unittest.main()
