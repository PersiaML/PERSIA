#!/usr/bin/env python3

import os
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

if __name__ == "__main__":
    import colorama
    colorama.init(autoreset=True)
    cwd = os.path.dirname(os.path.abspath(__file__))

    setup(
        name="persia-core",
        use_scm_version={"local_scheme": "no-local-version"},
        setup_requires=["setuptools_scm"],
        url="https://github.com/PersiaML/PersiaML-server",
        python_requires=">=3.6",
        description="Core Python binding for PersiaML.",
        package_dir={"": "python/"},
        packages=find_packages("python/"),
        rust_extensions=[
            RustExtension(
                "persia_core.persia_core",
                path="persia-core/Cargo.toml",
                binding=Binding.PyO3,
                native=True,
            )
        ],
        author="Kuaishou AI Platform",
        author_email="admin@mail.xrlian.com",
        install_requires=[],
        zip_safe=False,
    )
