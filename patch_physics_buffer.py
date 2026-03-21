import re

with open("core/physics/physics_buffer.py", "r") as f:
    content = f.read()

# I had replaced element_offsets but the fields were actually left as element_offsets: NDArray...
# Let's fix PhysicsBuffer to actually use ElementCSR
old_fields = """    element_offsets: NDArray[Index]
    element_Z: NDArray[Charge]
    element_fraction: NDArray[Float]"""

new_fields = """    element_csr: ElementCSR"""

content = content.replace(old_fields, new_fields)

with open("core/physics/physics_buffer.py", "w") as f:
    f.write(content)
