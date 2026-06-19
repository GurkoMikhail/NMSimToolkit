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

    @patch('socket.socket')
    def test_init_success(self, mock_socket_cls):
        mock_server = MagicMock()
        mock_socket_cls.return_value = mock_server

        handler = LiveTrajectoryHandler(port=5555, debug_mode=True)

        # Verify socket setup
        mock_socket_cls.assert_called_once()
        import socket
        mock_server.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        mock_server.bind.assert_called_once_with(('', 5555))
        mock_server.listen.assert_called_once_with(1)
        mock_server.setblocking.assert_called_once_with(False)

        self.assertEqual(handler.max_trajectories, 10000)
        self.assertEqual(handler.metadata.shape, (2,))
        self.assertEqual(handler.metadata.dtype, np.int64)
        self.assertIsNone(handler.client_socket)

    @patch('socket.socket')
    def test_process_trajectories_zero_copy(self, mock_socket_cls):
        mock_server = MagicMock()
        mock_client = MagicMock()
        mock_server.accept.return_value = (mock_client, ('127.0.0.1', 12345))
        mock_socket_cls.return_value = mock_server

        handler = LiveTrajectoryHandler(port=5555, debug_mode=True, max_trajectories=5)
        handler.step = 42

        pos_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        pos_y = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=np.float64)
        pos_z = np.array([1.2, 2.2, 3.2, 4.2, 5.2], dtype=np.float64)
        track_ids = np.array([10, 20, 30, 40, 50], dtype=np.uint64)

        chunk = {
            'type': 'interactions',
            'data': {
                'pos_x': pos_x,
                'pos_y': pos_y,
                'pos_z': pos_z,
                'particle_ID': track_ids
            }
        }

        handler.process_chunk(chunk)

        # Trigger sending by passing a dead_particles chunk
        dead_chunk = {
            'type': 'dead_particles',
            'data': np.array([], dtype=np.uint64)
        }
        handler.process_chunk(dead_chunk)

        # Verify accept
        mock_server.accept.assert_called_once()
        mock_client.settimeout.assert_called_once_with(0.05)

        # Metadata check
        self.assertEqual(handler.metadata[0], 42)
        self.assertEqual(handler.metadata[1], 5) # active_count is len(pos_x) -> 5
        self.assertEqual(handler.step, 43)

        self.assertEqual(mock_client.sendall.call_count, 5)

        calls = mock_client.sendall.call_args_list

        # 1. Metadata
        args, _ = calls[0]
        self.assertIsInstance(args[0], memoryview)
        self.assertTrue(np.array_equal(np.frombuffer(args[0], dtype=np.int64), [42, 5]))

        # 2. X
        args, _ = calls[1]
        self.assertIsInstance(args[0], memoryview)
        self.assertTrue(np.array_equal(np.frombuffer(args[0], dtype=np.float64), [1.0, 2.0, 3.0, 4.0, 5.0]))

        # 3. Y
        args, _ = calls[2]
        self.assertIsInstance(args[0], memoryview)
        self.assertTrue(np.array_equal(np.frombuffer(args[0], dtype=np.float64), [1.1, 2.1, 3.1, 4.1, 5.1]))

        # 4. Z
        args, _ = calls[3]
        self.assertIsInstance(args[0], memoryview)
        self.assertTrue(np.array_equal(np.frombuffer(args[0], dtype=np.float64), [1.2, 2.2, 3.2, 4.2, 5.2]))

        # 5. track_ids
        args, _ = calls[4]
        self.assertIsInstance(args[0], memoryview)
        self.assertTrue(np.array_equal(np.frombuffer(args[0], dtype=np.uint64), [10, 20, 30, 40, 50]))

    @patch('socket.socket')
    def test_process_trajectories_culling(self, mock_socket_cls):
        mock_server = MagicMock()
        mock_client = MagicMock()
        mock_server.accept.return_value = (mock_client, ('127.0.0.1', 12345))
        mock_socket_cls.return_value = mock_server

        # Limit to 2 trajectories
        handler = LiveTrajectoryHandler(port=5555, debug_mode=True, max_trajectories=2)
        handler.step = 10

        pos_x = np.arange(10, dtype=np.float64)
        pos_y = np.arange(10, dtype=np.float64)
        pos_z = np.arange(10, dtype=np.float64)
        track_ids = np.arange(10, dtype=np.uint64)

        chunk = {
            'type': 'interactions',
            'data': {
                'pos_x': pos_x,
                'pos_y': pos_y,
                'pos_z': pos_z,
                'particle_ID': track_ids
            }
        }

        handler.process_chunk(chunk)

        # Send dead particles chunk
        dead_chunk = {
            'type': 'dead_particles',
            'data': np.arange(5, 10, dtype=np.uint64)
        }
        handler.process_chunk(dead_chunk)

        # The first chunk had 10 points. We only had space for 2 points total.
        # It should cap at 2 due to cursor space limitation.
        self.assertEqual(handler.metadata[1], 2) # Should be culled to 2
        self.assertEqual(handler.step, 11)

        calls = mock_client.sendall.call_args_list
        args, _ = calls[1]
        self.assertEqual(len(np.frombuffer(args[0], dtype=np.float64)), 2)

    @patch('socket.socket')
    def test_process_trajectories_no_active(self, mock_socket_cls):
        mock_server = MagicMock()
        mock_client = MagicMock()
        mock_server.accept.return_value = (mock_client, ('127.0.0.1', 12345))
        mock_socket_cls.return_value = mock_server

        handler = LiveTrajectoryHandler(port=5555, debug_mode=True)

        # 0 active particles
        chunk = {
            'type': 'interactions',
            'data': {
                'pos_x': np.array([], dtype=np.float64),
                'pos_y': np.array([], dtype=np.float64),
                'pos_z': np.array([], dtype=np.float64),
                'particle_ID': np.array([], dtype=np.uint64)
            }
        }
        handler.process_chunk(chunk)

        dead_chunk = {
            'type': 'dead_particles',
            'data': np.array([], dtype=np.uint64)
        }
        handler.process_chunk(dead_chunk)

        # sendall should not be called because cursor is 0
        mock_client.sendall.assert_not_called()

    @patch('socket.socket')
    def test_client_drop_on_error(self, mock_socket_cls):
        mock_server = MagicMock()
        mock_client = MagicMock()
        mock_server.accept.return_value = (mock_client, ('127.0.0.1', 12345))
        mock_socket_cls.return_value = mock_server

        handler = LiveTrajectoryHandler(port=5555, debug_mode=True, max_trajectories=5)

        pos_x = np.array([1.0], dtype=np.float64)
        chunk = {
            'type': 'interactions',
            'data': {
                'pos_x': pos_x,
                'pos_y': pos_x,
                'pos_z': pos_x,
                'particle_ID': np.array([10], dtype=np.uint64)
            }
        }
        handler.process_chunk(chunk)

        # Simulate connection error
        mock_client.sendall.side_effect = ConnectionError("Connection dropped")

        dead_chunk = {'type': 'dead_particles', 'data': np.array([], dtype=np.uint64)}
        handler.process_chunk(dead_chunk)

        # Client should be closed and removed
        mock_client.close.assert_called_once()
        self.assertIsNone(handler.client_socket)


if __name__ == '__main__':
    unittest.main()
