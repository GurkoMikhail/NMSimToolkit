import re

with open("tests/transport/test_step5_transport.py", "r") as f:
    content = f.read()

# Try allocating NP empty array outside the kernel for test just to see if np.empty fails
content = content.replace("            transport_kernel(", "            # Let's print something \n            transport_kernel(")

with open("tests/transport/test_step5_transport.py", "w") as f:
    f.write(content)
