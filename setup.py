import os
import colorama

from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

use_cuda = os.environ.get("USE_CUDA", False)
native = os.environ.get("NATIVE", False)

if __name__ == "__main__":

    colorama.init(autoreset=True)

    features = None if not use_cuda else ["cuda"]
    rust_extensions = []

    rust_extensions.append(
        RustExtension(
            # TODO: Due to this issue https://github.com/PyO3/setuptools-rust/issues/153 still not release
            # the new version of setuptool_rust, RustExtension can't enable the script feature
            # {
            #     "persia-embedding-worker": "persia.persia_embedding_worker",
            #     "persia-embedding-parameter-server": "persia.persia_embedding_parameter_server"
            # },
            # script=True,
            {
                "persia-embedding-worker": "persia.persia-embedding-worker",
                "persia-embedding-parameter-server": "persia.persia-embedding-parameter-server",
            },
            path="rust/persia-embedding-server/Cargo.toml",
            binding=Binding.Exec,
            native=native,
        )
    )

    rust_extensions.append(
        RustExtension(
            "persia_core",
            path="rust/persia-core/Cargo.toml",
            binding=Binding.PyO3,
            native=native,
            features=features,
        )
    )

    rust_extensions.append(
        RustExtension(
            {
                "gencrd": "persia.gencrd",
                "operator": "persia.operator",
                "server": "persia.server",
            },
            path="k8s/Cargo.toml",
            binding=Binding.Exec,
            native=native,
        )
    )

    install_requires = ["colorlog", "pyyaml", "click"]

    name_suffix = os.getenv("PERSIA_CUDA_VERSION", "")
    if name_suffix != "":
        name_suffix = "-cuda" + name_suffix

    with open(os.path.realpath(os.path.join(__file__, "../README.md"))) as file:
        long_description = file.read()

    setup(
        name="persia" + name_suffix,
        use_scm_version={"local_scheme": "no-local-version"},
        setup_requires=["setuptools_scm"],
        install_requires=install_requires,
        url="https://github.com/PersiaML/PersiaML",
        author="Kuaishou AI Platform Persia Team",
        author_email="admin@mail.xrlian.com",
        description="PersiaML Python Library",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=("tests",)),
        rust_extensions=rust_extensions,
        entry_points={
            "console_scripts": [
                "persia-launcher=persia.launcher:cli",
                "persia-k8s-utils=persia.k8s_utils:cli",
            ]
        },
        python_requires=">=3.6",
    )