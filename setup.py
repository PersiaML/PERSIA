#!/usr/bin/env python3

import os
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

if __name__ == "__main__":
    import colorama
    colorama.init(autoreset=True)
    cwd = os.path.dirname(os.path.abspath(__file__))

    setup(
        name="persia_embedding_py_client_sharded_server",
        use_scm_version={"local_scheme": "no-local-version"},
        setup_requires=["setuptools_scm"],
        url="https://github.com/PersiaML/PersiaML-server",
        python_requires=">=3.6",
        description="Core Python binding for PersiaML.",
        package_dir={"": "python/"},
        packages=find_packages("python/"),
        rust_extensions=[
            RustExtension(
                "persia_embedding_py_client_sharded_server.persia_embedding_py_client_sharded_server",
                path="persia-embedding-py-client-sharded-server/Cargo.toml",
                binding=Binding.PyO3,
                native=True,
            )
        ],
        author="Kuaishou AI Platform",
        author_email="admin@mail.xrlian.com",
        install_requires=[],
        zip_safe=False,
    )
