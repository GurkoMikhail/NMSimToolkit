with open("tests/other/test_live_trajectory_handler.py", "r") as f:
    content = f.read()

import re
old_test_part1 = """        # 1. Metadata
        args, kwargs = calls[0]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)"""

new_test_part1 = """        # 1. Metadata
        args, kwargs = calls[0]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)
        self.assertFalse(kwargs['copy'])"""

content = content.replace(old_test_part1, new_test_part1)

old_test_part2 = """        # 2. X
        args, kwargs = calls[1]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)"""

new_test_part2 = """        # 2. X
        args, kwargs = calls[1]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)
        self.assertFalse(kwargs['copy'])"""

content = content.replace(old_test_part2, new_test_part2)

old_test_part3 = """        # 3. Y
        args, kwargs = calls[2]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)"""

new_test_part3 = """        # 3. Y
        args, kwargs = calls[2]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)
        self.assertFalse(kwargs['copy'])"""

content = content.replace(old_test_part3, new_test_part3)

old_test_part4 = """        # 4. Z
        args, kwargs = calls[3]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)"""

new_test_part4 = """        # 4. Z
        args, kwargs = calls[3]
        self.assertIsInstance(args[0], memoryview)
        self.assertEqual(kwargs['flags'], zmq.SNDMORE)
        self.assertFalse(kwargs['copy'])"""

content = content.replace(old_test_part4, new_test_part4)

old_test_part5 = """        # 5. track_ids
        args, kwargs = calls[4]
        self.assertIsInstance(args[0], memoryview)
        self.assertNotIn('flags', kwargs)"""

new_test_part5 = """        # 5. track_ids
        args, kwargs = calls[4]
        self.assertIsInstance(args[0], memoryview)
        self.assertNotIn('flags', kwargs)
        self.assertFalse(kwargs['copy'])"""

content = content.replace(old_test_part5, new_test_part5)

with open("tests/other/test_live_trajectory_handler.py", "w") as f:
    f.write(content)
