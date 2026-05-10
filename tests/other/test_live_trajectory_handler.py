import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from core.data.data_handlers import LiveTrajectoryHandler

class TestLiveTrajectoryHandler(unittest.TestCase):

    def test_init_raises_value_error_on_invalid_debug_mode(self):
        with self.assertRaisesRegex(ValueError, "LiveTrajectoryHandler requires debug_mode=True"):
            LiveTrajectoryHandler(port=5555, debug_mode=False)

    def test_init_raises_value_error_on_invalid_port(self):
        with self.assertRaisesRegex(ValueError, "LiveTrajectoryHandler requires a strictly positive port number"):
            LiveTrajectoryHandler(port=-1, debug_mode=True)

        with self.assertRaisesRegex(ValueError, "LiveTrajectoryHandler requires a strictly positive port number"):
            LiveTrajectoryHandler(port=0, debug_mode=True)

    @patch('zmq.Context')
    def test_init_success(self, mock_context):
        mock_socket = MagicMock()
        mock_context.return_value.socket.return_value = mock_socket

        handler = LiveTrajectoryHandler(port=5555, debug_mode=True)

        # Verify ZMQ setup
        mock_context.return_value.socket.assert_called_once()
        import zmq
        mock_socket.setsockopt.assert_called_once_with(zmq.SNDHWM, 10)
        mock_socket.bind.assert_called_once_with("tcp://*:5555")

        self.assertEqual(handler.max_trajectories, 10000)
        self.assertEqual(handler.metadata.shape, (2,))
        self.assertEqual(handler.metadata.dtype, np.int64)

    @patch('zmq.Context')
    def test_process_trajectories_zero_copy(self, mock_context):
        mock_socket = MagicMock()
        mock_context.return_value.socket.return_value = mock_socket

        handler = LiveTrajectoryHandler(port=5555, debug_mode=True, max_trajectories=5)

        step = 42
        active_count = 3
        pos_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        pos_y = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=np.float64)
        pos_z = np.array([1.2, 2.2, 3.2, 4.2, 5.2], dtype=np.float64)
        track_ids = np.array([10, 20, 30, 40, 50], dtype=np.uint64)

        handler.process_trajectories(step, active_count, pos_x, pos_y, pos_z, track_ids)

        # Metadata check
        self.assertEqual(handler.metadata[0], 42)
        self.assertEqual(handler.metadata[1], 3)

        self.assertEqual(mock_socket.send.call_count, 5)

        # We need to verify that what's being sent is a memoryview without copies.
        # MagicMock records arguments.
        calls = mock_socket.send.call_args_list
        import zmq

        # 1. Metadata
        args, kwargs = calls[0]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)
        self.assertFalse(kwargs['copy'])
        self.assertTrue(np.array_equal(np.frombuffer(args[0], dtype=np.int64), [42, 3]))

        # 2. X
        args, kwargs = calls[1]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)
        self.assertFalse(kwargs['copy'])
        self.assertTrue(np.array_equal(np.frombuffer(args[0], dtype=np.float64), [1.0, 2.0, 3.0]))

        # 3. Y
        args, kwargs = calls[2]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)
        self.assertFalse(kwargs['copy'])
        self.assertTrue(np.array_equal(np.frombuffer(args[0], dtype=np.float64), [1.1, 2.1, 3.1]))

        # 4. Z
        args, kwargs = calls[3]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)
        self.assertFalse(kwargs['copy'])
        self.assertTrue(np.array_equal(np.frombuffer(args[0], dtype=np.float64), [1.2, 2.2, 3.2]))

        # 5. track_ids
        args, kwargs = calls[4]
        self.assertIsInstance(args[0], memoryview)
        self.assertNotIn('flags', kwargs)
        self.assertFalse(kwargs['copy'])
        self.assertTrue(np.array_equal(np.frombuffer(args[0], dtype=np.uint64), [10, 20, 30]))

    @patch('zmq.Context')
    def test_process_trajectories_culling(self, mock_context):
        mock_socket = MagicMock()
        mock_context.return_value.socket.return_value = mock_socket

        # Limit to 2 trajectories
        handler = LiveTrajectoryHandler(port=5555, debug_mode=True, max_trajectories=2)

        step = 10
        active_count = 5 # more than max_trajectories
        pos_x = np.arange(10, dtype=np.float64)
        pos_y = np.arange(10, dtype=np.float64)
        pos_z = np.arange(10, dtype=np.float64)
        track_ids = np.arange(10, dtype=np.uint64)

        handler.process_trajectories(step, active_count, pos_x, pos_y, pos_z, track_ids)

        self.assertEqual(handler.metadata[1], 2) # Should be culled to 2
        calls = mock_socket.send.call_args_list

        # check that only 2 elements were passed
        args, _ = calls[1]
        self.assertEqual(len(np.frombuffer(args[0], dtype=np.float64)), 2)

    @patch('zmq.Context')
    def test_process_trajectories_no_active(self, mock_context):
        mock_socket = MagicMock()
        mock_context.return_value.socket.return_value = mock_socket

        handler = LiveTrajectoryHandler(port=5555, debug_mode=True)

        # 0 active particles
        handler.process_trajectories(0, 0, np.array([]), np.array([]), np.array([]), np.array([]))

        # send should not be called
        mock_socket.send.assert_not_called()

if __name__ == '__main__':
    unittest.main()
