import os
import tempfile
import unittest
from typing import Dict, List, Mapping

import h5py
import numpy as np

from core.data.dataset_write_buffer import DatasetLayout, DatasetWriteBuffer

GROUP_STATES = "initial_states"
GROUP_INTERACTIONS = "interactions/Detector_1"

# Байт на строку: float32 (N, 3) = 12, uint8 (N,) = 1, uint64 (N,) = 8
ROW_BYTES = 12 + 1 + 8
ATTRS: Mapping[str, Mapping[str, str]] = {
    "position": {"units": "mm"},
    "code": {"mapping": "0=a,1=b"},
}


def make_portion(rng: np.random.Generator, row_count: int) -> Dict[str, np.ndarray]:
    """Порция с тремя полями разных dtype/row_shape со случайным содержимым."""
    return {
        "position": rng.random((row_count, 3), dtype=np.float32),
        "code": rng.integers(0, 255, size=row_count, dtype=np.uint8),
        "identifier": rng.integers(0, 2**63, size=row_count, dtype=np.uint64),
    }


def concat_portions(portions: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Эталон: конкатенация порций по полям в порядке добавления."""
    return {
        field_name: np.concatenate([portion[field_name] for portion in portions], axis=0)
        for field_name in portions[0]
    }


class TestDatasetWriteBuffer(unittest.TestCase):
    """Модульные тесты DatasetWriteBuffer на реальных HDF5-файлах."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.h5_path = os.path.join(self.temp_dir.name, "buffer_test.h5")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def assert_group_equals(self, h5_file: h5py.File, group_path: str, expected: Dict[str, np.ndarray]) -> None:
        """Побайтово сверяет все поля группы с эталоном."""
        for field_name, expected_array in expected.items():
            actual = h5_file[f"{group_path}/{field_name}"][...]
            self.assertEqual(actual.dtype, expected_array.dtype, field_name)
            self.assertEqual(actual.shape, expected_array.shape, field_name)
            self.assertEqual(actual.tobytes(), expected_array.tobytes(), field_name)

    def group_lengths(self, h5_file: h5py.File, group_path: str, field_names: List[str]) -> List[int]:
        return [int(h5_file[f"{group_path}/{field_name}"].shape[0]) for field_name in field_names]

    def test_constructor_rejects_non_positive_buffer_size(self) -> None:
        with self.assertRaises(ValueError):
            DatasetWriteBuffer(0, None, 0)
        with self.assertRaises(ValueError):
            DatasetWriteBuffer(-5, "gzip", 4)

    def test_layout_row_bytes(self) -> None:
        layout = DatasetLayout(np.dtype(np.float32), (3,), {})
        self.assertEqual(layout.row_bytes, 12)
        self.assertEqual(DatasetLayout(np.dtype(np.uint64), (), {}).row_bytes, 8)

    def test_chunk_rows_formula(self) -> None:
        rng = np.random.default_rng(1)
        buffer = DatasetWriteBuffer(ROW_BYTES * 100 + 5, None, 0)
        buffer.append(GROUP_STATES, make_portion(rng, 3), ATTRS)
        self.assertEqual(buffer.chunk_rows(GROUP_STATES), 100)

        tiny = DatasetWriteBuffer(1, None, 0)
        tiny.append(GROUP_STATES, make_portion(rng, 3), ATTRS)
        self.assertEqual(tiny.chunk_rows(GROUP_STATES), 1)

        with self.assertRaises(KeyError):
            buffer.chunk_rows("unknown")

    def test_portions_below_chunk_do_not_write(self) -> None:
        rng = np.random.default_rng(2)
        buffer = DatasetWriteBuffer(ROW_BYTES * 50, None, 0)
        buffer.append(GROUP_STATES, make_portion(rng, 10), ATTRS)
        buffer.append(GROUP_STATES, make_portion(rng, 39), ATTRS)
        self.assertFalse(buffer.has_full_chunks)
        self.assertEqual(buffer.pending_rows(GROUP_STATES), 49)
        with h5py.File(self.h5_path, "w") as h5_file:
            buffer.write_full_chunks(h5_file)
            self.assertEqual(list(h5_file.keys()), [])
        self.assertEqual(buffer.pending_rows(GROUP_STATES), 49)

    def test_write_all_equals_concatenation_two_groups(self) -> None:
        rng = np.random.default_rng(3)
        buffer = DatasetWriteBuffer(ROW_BYTES * 40, None, 0)
        states_portions = [make_portion(rng, size) for size in (3, 17, 1, 60, 5)]
        interactions_portions = [make_portion(rng, size) for size in (7, 2, 90)]
        for portion in states_portions:
            buffer.append(GROUP_STATES, portion, ATTRS)
        for portion in interactions_portions:
            buffer.append(GROUP_INTERACTIONS, portion, ATTRS)
        self.assertTrue(buffer.has_full_chunks)

        with h5py.File(self.h5_path, "w") as h5_file:
            buffer.write_all(h5_file)
        self.assertEqual(buffer.pending_rows(GROUP_STATES), 0)
        self.assertEqual(buffer.pending_rows(GROUP_INTERACTIONS), 0)
        self.assertFalse(buffer.has_full_chunks)

        with h5py.File(self.h5_path, "r") as h5_file:
            self.assert_group_equals(h5_file, GROUP_STATES, concat_portions(states_portions))
            self.assert_group_equals(h5_file, GROUP_INTERACTIONS, concat_portions(interactions_portions))

    def test_chunks_independent_of_first_portion_and_attrs_set(self) -> None:
        rng = np.random.default_rng(4)
        buffer_size = ROW_BYTES * 50_000
        buffer = DatasetWriteBuffer(buffer_size, None, 0)
        portion = make_portion(rng, 3)
        buffer.append(GROUP_STATES, portion, ATTRS)
        with h5py.File(self.h5_path, "w") as h5_file:
            buffer.write_all(h5_file)
        with h5py.File(self.h5_path, "r") as h5_file:
            expected_chunk_rows = buffer_size // ROW_BYTES
            for field_name in portion:
                dataset = h5_file[f"{GROUP_STATES}/{field_name}"]
                self.assertEqual(dataset.chunks[0], expected_chunk_rows)
                self.assertEqual(dataset.shape[0], 3)
                self.assertEqual(dataset.maxshape[0], None)
                self.assertIsNone(dataset.compression)
            self.assertEqual(h5_file[f"{GROUP_STATES}/position"].chunks, (expected_chunk_rows, 3))
            self.assertEqual(h5_file[f"{GROUP_STATES}/position"].attrs["units"], "mm")
            self.assertEqual(h5_file[f"{GROUP_STATES}/code"].attrs["mapping"], "0=a,1=b")
            self.assertNotIn("units", h5_file[f"{GROUP_STATES}/identifier"].attrs)

    def test_incremental_flush_alignment_property(self) -> None:
        for seed in range(5):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                chunk_rows = int(rng.integers(5, 60))
                buffer = DatasetWriteBuffer(ROW_BYTES * chunk_rows, None, 0)
                portions: List[Dict[str, np.ndarray]] = []
                field_names = ["position", "code", "identifier"]
                path = os.path.join(self.temp_dir.name, f"property_{seed}.h5")
                with h5py.File(path, "w") as h5_file:
                    for _ in range(120):
                        portion = make_portion(rng, int(rng.integers(0, 30)))
                        portions.append(portion)
                        buffer.append(GROUP_STATES, portion, ATTRS)
                        buffer.write_full_chunks(h5_file)
                        if GROUP_STATES in h5_file:
                            lengths = self.group_lengths(h5_file, GROUP_STATES, field_names)
                            self.assertEqual(len(set(lengths)), 1)
                            self.assertEqual(lengths[0] % chunk_rows, 0)
                            self.assertEqual(lengths[0] + buffer.pending_rows(GROUP_STATES),
                                             sum(len(item["code"]) for item in portions))
                            self.assertLess(buffer.pending_rows(GROUP_STATES), chunk_rows)
                    buffer.write_all(h5_file)
                with h5py.File(path, "r") as h5_file:
                    self.assert_group_equals(h5_file, GROUP_STATES, concat_portions(portions))

    def test_resume_completes_tail_chunk(self) -> None:
        rng = np.random.default_rng(5)
        chunk_rows = 50
        buffer_size = ROW_BYTES * chunk_rows
        field_names = ["position", "code", "identifier"]

        first_run = [make_portion(rng, 70)]
        with h5py.File(self.h5_path, "w") as h5_file:
            buffer = DatasetWriteBuffer(buffer_size, None, 0)
            buffer.append(GROUP_STATES, first_run[0], ATTRS)
            buffer.write_all(h5_file)
            self.assertEqual(self.group_lengths(h5_file, GROUP_STATES, field_names), [70, 70, 70])

        second_run: List[Dict[str, np.ndarray]] = []
        with h5py.File(self.h5_path, "a") as h5_file:
            buffer = DatasetWriteBuffer(buffer_size, None, 0)
            portion = make_portion(rng, 10)
            second_run.append(portion)
            buffer.append(GROUP_STATES, portion, ATTRS)
            buffer.write_full_chunks(h5_file)
            self.assertEqual(self.group_lengths(h5_file, GROUP_STATES, field_names), [70, 70, 70])

            portion = make_portion(rng, 45)
            second_run.append(portion)
            buffer.append(GROUP_STATES, portion, ATTRS)
            self.assertTrue(buffer.has_full_chunks)
            buffer.write_full_chunks(h5_file)
            # 70 + 55 = 125 -> выравнивание до 100: дозаполнен хвостовой чанк, 25 строк остаются в буфере
            self.assertEqual(self.group_lengths(h5_file, GROUP_STATES, field_names), [100, 100, 100])
            self.assertEqual(buffer.pending_rows(GROUP_STATES), 25)

            for _ in range(4):
                portion = make_portion(rng, 33)
                second_run.append(portion)
                buffer.append(GROUP_STATES, portion, ATTRS)
                buffer.write_full_chunks(h5_file)
                lengths = self.group_lengths(h5_file, GROUP_STATES, field_names)
                self.assertEqual(len(set(lengths)), 1)
                self.assertEqual(lengths[0] % chunk_rows, 0)
            buffer.write_all(h5_file)

        with h5py.File(self.h5_path, "r") as h5_file:
            self.assert_group_equals(h5_file, GROUP_STATES, concat_portions(first_run + second_run))

    def test_resume_with_changed_buffer_size_uses_existing_chunks(self) -> None:
        rng = np.random.default_rng(6)
        field_names = ["position", "code", "identifier"]
        first_run = [make_portion(rng, 25)]
        with h5py.File(self.h5_path, "w") as h5_file:
            buffer = DatasetWriteBuffer(ROW_BYTES * 50, None, 0)
            buffer.append(GROUP_STATES, first_run[0], ATTRS)
            buffer.write_all(h5_file)

        second_run: List[Dict[str, np.ndarray]] = []
        with h5py.File(self.h5_path, "a") as h5_file:
            buffer = DatasetWriteBuffer(ROW_BYTES * 10, None, 0)  # собственный chunk_rows = 10
            portion = make_portion(rng, 40)
            second_run.append(portion)
            buffer.append(GROUP_STATES, portion, ATTRS)
            self.assertEqual(buffer.chunk_rows(GROUP_STATES), 10)
            buffer.write_full_chunks(h5_file)
            # используется chunks[0] = 50 существующих датасетов: 25 + 40 -> 50, 15 остаётся
            self.assertEqual(buffer.chunk_rows(GROUP_STATES), 50)
            self.assertEqual(self.group_lengths(h5_file, GROUP_STATES, field_names), [50, 50, 50])
            self.assertEqual(buffer.pending_rows(GROUP_STATES), 15)
            self.assertFalse(buffer.has_full_chunks)
            for field_name in field_names:
                self.assertEqual(h5_file[f"{GROUP_STATES}/{field_name}"].chunks[0], 50)
            buffer.write_all(h5_file)

        with h5py.File(self.h5_path, "r") as h5_file:
            self.assert_group_equals(h5_file, GROUP_STATES, concat_portions(first_run + second_run))

    def test_append_validation_errors(self) -> None:
        rng = np.random.default_rng(7)
        buffer = DatasetWriteBuffer(ROW_BYTES * 50, None, 0)
        buffer.append(GROUP_STATES, make_portion(rng, 5), ATTRS)

        wrong_dtype = make_portion(rng, 4)
        wrong_dtype["code"] = wrong_dtype["code"].astype(np.int16)
        with self.assertRaises(ValueError):
            buffer.append(GROUP_STATES, wrong_dtype, ATTRS)

        missing_field = make_portion(rng, 4)
        del missing_field["code"]
        with self.assertRaises(ValueError):
            buffer.append(GROUP_STATES, missing_field, ATTRS)

        extra_field = make_portion(rng, 4)
        extra_field["extra"] = np.zeros(4, dtype=np.uint8)
        with self.assertRaises(ValueError):
            buffer.append(GROUP_STATES, extra_field, ATTRS)

        wrong_row_shape = make_portion(rng, 4)
        wrong_row_shape["position"] = rng.random((4, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            buffer.append(GROUP_STATES, wrong_row_shape, ATTRS)

        different_lengths = make_portion(rng, 4)
        different_lengths["code"] = different_lengths["code"][:3]
        with self.assertRaises(ValueError):
            buffer.append(GROUP_STATES, different_lengths, ATTRS)

        with self.assertRaises(ValueError):
            DatasetWriteBuffer(100, None, 0).append("new_group", different_lengths, ATTRS)
        with self.assertRaises(ValueError):
            DatasetWriteBuffer(100, None, 0).append("new_group", {}, ATTRS)
        with self.assertRaises(ValueError):
            DatasetWriteBuffer(100, None, 0).append("new_group", {"scalar": np.float32(1.0)}, ATTRS)

        # Отклонённые порции не изменили состояние буфера
        self.assertEqual(buffer.pending_rows(GROUP_STATES), 5)

    def test_empty_portion_changes_nothing(self) -> None:
        rng = np.random.default_rng(8)
        buffer = DatasetWriteBuffer(ROW_BYTES * 50, None, 0)
        buffer.append(GROUP_STATES, make_portion(rng, 0), ATTRS)
        with self.assertRaises(KeyError):
            buffer.chunk_rows(GROUP_STATES)
        self.assertEqual(buffer.pending_rows(GROUP_STATES), 0)
        with h5py.File(self.h5_path, "w") as h5_file:
            buffer.write_all(h5_file)
            self.assertEqual(list(h5_file.keys()), [])

        portions = [make_portion(rng, 4), make_portion(rng, 0), make_portion(rng, 6)]
        for portion in portions:
            buffer.append(GROUP_STATES, portion, ATTRS)
        self.assertEqual(buffer.pending_rows(GROUP_STATES), 10)
        with h5py.File(self.h5_path, "w") as h5_file:
            buffer.write_all(h5_file)
        with h5py.File(self.h5_path, "r") as h5_file:
            self.assert_group_equals(h5_file, GROUP_STATES, concat_portions(portions))

        # Пустая порция с неверной схемой в известной группе отклоняется
        wrong_empty = make_portion(rng, 0)
        wrong_empty["code"] = wrong_empty["code"].astype(np.int16)
        with self.assertRaises(ValueError):
            buffer.append(GROUP_STATES, wrong_empty, ATTRS)

    def test_gzip_compression_applied(self) -> None:
        rng = np.random.default_rng(9)
        buffer = DatasetWriteBuffer(ROW_BYTES * 20, "gzip", 4)
        portion = make_portion(rng, 30)
        buffer.append(GROUP_STATES, portion, ATTRS)
        with h5py.File(self.h5_path, "w") as h5_file:
            buffer.write_all(h5_file)
        with h5py.File(self.h5_path, "r") as h5_file:
            for field_name in portion:
                dataset = h5_file[f"{GROUP_STATES}/{field_name}"]
                self.assertEqual(dataset.compression, "gzip")
                self.assertEqual(dataset.compression_opts, 4)
                self.assertEqual(dataset.chunks[0], 20)
            self.assert_group_equals(h5_file, GROUP_STATES, concat_portions([portion]))

    def test_corrupted_file_unequal_lengths_raises(self) -> None:
        rng = np.random.default_rng(10)
        with h5py.File(self.h5_path, "w") as h5_file:
            buffer = DatasetWriteBuffer(ROW_BYTES * 20, None, 0)
            buffer.append(GROUP_STATES, make_portion(rng, 30), ATTRS)
            buffer.write_all(h5_file)
            h5_file[f"{GROUP_STATES}/code"].resize(29, axis=0)

            resumed = DatasetWriteBuffer(ROW_BYTES * 20, None, 0)
            resumed.append(GROUP_STATES, make_portion(rng, 30), ATTRS)
            with self.assertRaisesRegex(RuntimeError, "различаются"):
                resumed.write_full_chunks(h5_file)
            with self.assertRaisesRegex(RuntimeError, "различаются"):
                resumed.write_all(h5_file)
            self.assertEqual(resumed.pending_rows(GROUP_STATES), 30)

    def test_existing_file_with_missing_dataset_or_wrong_dtype_raises(self) -> None:
        rng = np.random.default_rng(11)
        with h5py.File(self.h5_path, "w") as h5_file:
            buffer = DatasetWriteBuffer(ROW_BYTES * 20, None, 0)
            buffer.append(GROUP_STATES, make_portion(rng, 30), ATTRS)
            buffer.write_all(h5_file)

            del h5_file[f"{GROUP_STATES}/identifier"]
            resumed = DatasetWriteBuffer(ROW_BYTES * 20, None, 0)
            resumed.append(GROUP_STATES, make_portion(rng, 30), ATTRS)
            with self.assertRaisesRegex(RuntimeError, "неконсистентна"):
                resumed.write_all(h5_file)

            h5_file.create_dataset(
                f"{GROUP_STATES}/identifier", shape=(30,), maxshape=(None,), chunks=(20,), dtype=np.int32
            )
            with self.assertRaises(ValueError):
                resumed.write_all(h5_file)

    def test_buffer_preserves_fifo_without_copy_of_inputs(self) -> None:
        rng = np.random.default_rng(12)
        buffer = DatasetWriteBuffer(ROW_BYTES * 10, None, 0)
        portions = [make_portion(rng, 7) for _ in range(3)]
        for portion in portions:
            for array in portion.values():
                array.flags.writeable = False  # read-only-вью допустимы
            buffer.append(GROUP_STATES, portion, ATTRS)
        with h5py.File(self.h5_path, "w") as h5_file:
            buffer.write_full_chunks(h5_file)
            self.assertEqual(self.group_lengths(h5_file, GROUP_STATES, ["position", "code", "identifier"]), [20, 20, 20])
            self.assertEqual(buffer.pending_rows(GROUP_STATES), 1)
            buffer.write_all(h5_file)
        with h5py.File(self.h5_path, "r") as h5_file:
            self.assert_group_equals(h5_file, GROUP_STATES, concat_portions(portions))


if __name__ == "__main__":
    unittest.main()
