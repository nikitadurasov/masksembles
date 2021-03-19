import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="masksembles",
    version="1.1",
    author="Nikita Durasov",
    author_email="yassnda@gmail.com",
    description="Official implementation of Masksembles approach",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikitadurasov/masksembles",
    packages=setuptools.find_packages()
)
