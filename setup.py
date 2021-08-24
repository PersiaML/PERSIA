import os
import subprocess

from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

use_cuda = os.environ.get("USE_CUDA", False)
integration_core = os.environ.get("INTEGRATION_CORE", True)

if __name__ == "__main__":
    extensions = []
    cmdclass = {}

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
        cmdclass["build_ext"] = BuildExtension

    rust_extensions = []

    if integration_core:
        features = None if not use_cuda else ["cuda"]
        rust_extensions.append(
            RustExtension(
                "persia_core.persia_core",
                path="rust/persia-core/Cargo.toml",
                binding=Binding.PyO3,
                native=True,
                features=features,
            )
        )
        

    def get_mpi_flags():
        flags = subprocess.check_output("mpicxx -show", shell=True).decode().split()[1:]
        print(flags)
        return flags


    setup(
        name="persia",
        use_scm_version={"local_scheme": "no-local-version"},
        setup_requires=["setuptools_scm"],
        install_requires=["pyyaml", "persia-core", "colorlog"],
        url="https://github.com/PersiaML/PersiaML",
        author="Kuaishou AI Platform Persia Team",
        author_email="admin@mail.xrlian.com",
        description="PersiaML Python Library",
        packages=find_packages(exclude=("tests",)),
        scripts=["bin/launch_middleware", "bin/launch_server"],
        ext_modules=extensions,
        cmdclass=cmdclass,
        rust_extensions=rust_extensions,
        python_requires=">=3.7",
    )
