import re

with open("core/transport/transport_kernels.py", "r") as f:
    content = f.read()

content = content.replace(
    "real_material_id = call_cfunc_ptr",
    "material_id = call_cfunc_ptr"
)

with open("core/transport/transport_kernels.py", "w") as f:
    f.write(content)
