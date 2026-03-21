import re

with open("core/transport/transport_kernels.py", "r") as f:
    content = f.read()

# Fix call_cfunc_ptr to avoid Signature weakref bug in Numba.
# The `get_function_pointer_type` expects a Numba type (like types.ExternalFunctionPointer), not a Signature.
# Or we can just use cgutils.get_function(builder, ...) or manually create ir.FunctionType.

fixed_cfunc = """@intrinsic
def call_cfunc_ptr(typingctx, ptr, x, y, z):
    sig = types.int64(types.uint64, types.float64, types.float64, types.float64)
    def codegen(context, builder, signature, args):
        ptr_val, x_val, y_val, z_val = args
        # Cast integer pointer to a function pointer
        from llvmlite import ir
        fnty = ir.FunctionType(ir.IntType(64), [ir.DoubleType(), ir.DoubleType(), ir.DoubleType()])
        fnptr = builder.inttoptr(ptr_val, fnty.as_pointer())
        return builder.call(fnptr, [x_val, y_val, z_val])
    return sig, codegen
"""

content = re.sub(
    r'@intrinsic\ndef call_cfunc_ptr[\s\S]*?return sig, codegen\n',
    fixed_cfunc,
    content
)

with open("core/transport/transport_kernels.py", "w") as f:
    f.write(content)
