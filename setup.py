import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
readme = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="masksembles",                 # change if taken on PyPI
    version="1.1.1",                    # use semantic x.y.z (PyPI won’t accept re-uploads)
    author="Nikita Durasov",
    author_email="yassnda@gmail.com",
    description="Official implementation of Masksembles approach",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/nikitadurasov/masksembles",
    project_urls={
        "Issues": "https://github.com/nikitadurasov/masksembles/issues",
        "Documentation": "https://github.com/nikitadurasov/masksembles#readme",
    },
    license="MIT",
    packages=find_packages(exclude=("tests", "test", "notebooks")),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "torch": ["torch>=1.12"],
        "tensorflow": ["tensorflow>=2.9"],
        "dev": ["pytest", "black", "flake8"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=["uncertainty", "ensembles", "deep-learning", "pytorch", "tensorflow"],
)
