import os
import subprocess

from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

extensions, rust_extensions = [], []
cmdclass = {}

use_cuda = os.environ.get("USE_CUDA", False)

if use_cuda:
    import torch
    from torch.utils.cpp_extension import BuildExtension
    from torch.utils.cpp_extension import CUDAExtension

    extensions.append(
        CUDAExtension(
            "persia_torch_ext",
            include_dirs=[
                os.getcwd() + "/torch_ext/third_party/thread_pool",
                os.getcwd() + "/torch_ext/third_party/dbg_macro",
                "/opt/cuda/include",
            ],
            sources=[
                "torch_ext/persia_torch_ext.cpp",
            ],
            extra_compile_args=["-fopenmp"],
        ),
    )

    rust_extensions.append(
        RustExtension(
            "persia_embedding_py_client_sharded_server",
            path="persia-embedding-real/persia-embedding-py-client-sharded-server/Cargo.toml",
            features=["cuda"],
            binding=Binding.PyO3,
        )
    )
    cmdclass={"build_ext": BuildExtension}
else:
    rust_extensions.append(
        RustExtension(
            "persia_embedding_py_client_sharded_server",
            path="persia-embedding-real/persia-embedding-py-client-sharded-server/Cargo.toml",
            binding=Binding.PyO3,
        )
    )


def get_mpi_flags():
    flags = subprocess.check_output("mpicxx -show", shell=True).decode().split()[1:]
    print(flags)
    return flags


setup(
    name="persia",
    version="0.1.0",
    author="Kuaishou AI Platform Persia Team",
    author_email="admin@mail.xrlian.com",
    description="PersiaML Python Library",
    packages=find_packages(),
    scripts=["bin/launch_middleware", "bin/launch_server"],
    ext_modules=extensions,
    rust_extensions=rust_extensions,
    cmdclass=cmdclass,
    python_requires=">=3.7",
)
