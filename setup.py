import os

from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

use_cuda = os.environ.get("USE_CUDA", False)
integration_core = os.environ.get("INTEGRATION_CORE", True)

if __name__ == "__main__":
    import colorama

    colorama.init(autoreset=True)

    extensions = []
    cmdclass = {}

    if use_cuda:
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

    rust_extensions.append(
        RustExtension(
            # {
            #     "persia_ps.middleware": "persia-embedding-sharded-middleware",
            #     "persia_ps.server": "persia-embedding-sharded-server"
            # },
            {
                "persia-embedding-sharded-middleware": "persia_ps.middleware",
                "persia-embedding-sharded-server": "persia_ps.server"
            },
            path="rust/persia-embedding-sharded-server/Cargo.toml",
            binding=Binding.Exec,
            script=True,
            native=True,
        )
    )

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
        ext_modules=extensions,
        cmdclass=cmdclass,
        rust_extensions=rust_extensions,
        entry_points={
            "console_scripts": [
                "launch_trainer = persia.launcher:launch_trainer",
                "launch_compose = persia.launcher:launch_trainer",
                "launch_middleware = persia.launcher:launch_middleware",
                "launch_server = persia.launcher:launch_server",
                "launch_local = persia.launcher:launch_trainer",
            ]
        },
        python_requires=">=3.7",
    )
