#!/usr/bin/env python3

import os
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension


if __name__ == "__main__":
    import colorama
    colorama.init(autoreset=True)

    use_cuda = os.environ.get("USE_CUDA", False)
    features = None if not use_cuda else ["cuda"]

    setup(
        name="persia-core",
        use_scm_version={"local_scheme": "no-local-version", "root": "..", "relative_to": __file__},
        setup_requires=["setuptools_scm"],
        url="https://github.com/PersiaML/PersiaML/rust",
        python_requires=">=3.7",
        description="Core Python binding for PersiaML.",
        package_dir={"": "python/"},
        packages=find_packages("python/"),
        rust_extensions=[
            RustExtension(
                "persia_core.persia_core",
                path="persia-core/Cargo.toml",
                binding=Binding.PyO3,
                native=True,
                features=features,
            )
        ],
        author="Kuaishou AI Platform",
        author_email="admin@mail.xrlian.com",
        install_requires=[],
        zip_safe=False,
    )
