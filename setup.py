import os
from typing import List

from setuptools import find_packages, setup

_repo: str = "amazon-route-optimization"
_pkg: str = "aro"
_version = "0.0.1"


def read(fname) -> str:
    """Read the content of a file.

    You may use this to get the content of, for e.g., requirements.txt, VERSION, etc.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Declare minimal set for installation
setup(
    name=_pkg,
    packages=find_packages(where="."),
    version=_version,
    description="A Demo of the Amazon Last Mile Routing Research Challenge Solution",
    long_description=read("README.md"),
    author="AWS Professional Services",
    url=f"https://github.com/aws-samples/{_repo}/",
    download_url="",
    project_urls={
        "Bug Tracker": f"https://github.com/abcd/{_repo}/issues/",
        "Documentation": f"https://{_repo}.readthedocs.io/en/stable/",
        "Source Code": f"https://github.com/abcd/{_repo}/",
    },
    python_requires=">=3.6.0",
)
