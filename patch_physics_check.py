import re

with open("core/physics/physics_kernels.py", "r") as f:
    content = f.read()

# Let's see physics_kernels.py logic
print("Checking _get_macroscopic_cross_sections")
