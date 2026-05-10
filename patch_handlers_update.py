with open("core/data/data_handlers.py", "r") as f:
    content = f.read()

# Replace import inside process_trajectories and add copy=False
old_method = """    def process_trajectories(self, step: int, active_count: int,
                             pos_x: np.ndarray, pos_y: np.ndarray, pos_z: np.ndarray,
                             track_ids: np.ndarray) -> None:
        import zmq
        send_count = min(active_count, self.max_trajectories)
        if send_count <= 0:
            return

        self.metadata[0] = step
        self.metadata[1] = send_count

        # multipart-message ZMQ (SNDMORE): metadata -> X -> Y -> Z -> track_ids
        self.socket.send(memoryview(self.metadata), flags=zmq.SNDMORE)
        self.socket.send(memoryview(pos_x[:send_count]), flags=zmq.SNDMORE)
        self.socket.send(memoryview(pos_y[:send_count]), flags=zmq.SNDMORE)
        self.socket.send(memoryview(pos_z[:send_count]), flags=zmq.SNDMORE)
        self.socket.send(memoryview(track_ids[:send_count]))"""

new_method = """    def process_trajectories(self, step: int, active_count: int,
                             pos_x: np.ndarray, pos_y: np.ndarray, pos_z: np.ndarray,
                             track_ids: np.ndarray) -> None:
        send_count = min(active_count, self.max_trajectories)
        if send_count <= 0:
            return

        self.metadata[0] = step
        self.metadata[1] = send_count

        # multipart-message ZMQ (SNDMORE): metadata -> X -> Y -> Z -> track_ids
        # Use copy=False to guarantee zero-copy transfer
        self.socket.send(memoryview(self.metadata), flags=self._sndmore, copy=False)
        self.socket.send(memoryview(pos_x[:send_count]), flags=self._sndmore, copy=False)
        self.socket.send(memoryview(pos_y[:send_count]), flags=self._sndmore, copy=False)
        self.socket.send(memoryview(pos_z[:send_count]), flags=self._sndmore, copy=False)
        self.socket.send(memoryview(track_ids[:send_count]), copy=False)"""

content = content.replace(old_method, new_method)

# Replace init to cache sndmore and import at top
old_init = """    def __init__(self, port: int, debug_mode: bool, max_trajectories: int = 10000):
        super().__init__()
        if not debug_mode:
            raise ValueError("LiveTrajectoryHandler requires debug_mode=True")
        if port <= 0:
            raise ValueError("LiveTrajectoryHandler requires a strictly positive port number")

        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 10)
        self.socket.bind(f"tcp://*:{port}")

        self.max_trajectories = max_trajectories
        self.metadata = np.empty(2, dtype=np.int64)"""

new_init = """    def __init__(self, port: int, debug_mode: bool, max_trajectories: int = 10000):
        super().__init__()
        if not debug_mode:
            raise ValueError("LiveTrajectoryHandler requires debug_mode=True")
        if port <= 0:
            raise ValueError("LiveTrajectoryHandler requires a strictly positive port number")

        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 10)
        self.socket.bind(f"tcp://*:{port}")
        self._sndmore = zmq.SNDMORE

        self.max_trajectories = max_trajectories
        self.metadata = np.empty(2, dtype=np.int64)"""

content = content.replace(old_init, new_init)

with open("core/data/data_handlers.py", "w") as f:
    f.write(content)
