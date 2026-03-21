import re

with open("tests/transport/test_step5_transport.py", "r") as f:
    content = f.read()

content = content.replace("ParticleBank.allocate(self.capacity)", "ParticleBank(self.capacity)")

with open("tests/transport/test_step5_transport.py", "w") as f:
    f.write(content)
