from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="PyLorentz",
    version='1.0.1',
    packages=find_packages(),
    package_dir={'hipl': './hipl'},
    description="A codebase designed for analyzing Lorentz Transmission Electron Microscopy (LTEM) data",
    long_description=long_description,
    url="https://github.com/pylorentz/pylorentz",
    author="Arthur R. C. McCray",
    author_email="arthurmccray95@gmail.com",
    license="MIT",
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "ipympl",
        "jupyter",
        "scikit-image",
        "matplotlib",
        "ncempy",
        "colorcet",
        "black",
        "tqdm",
    ],
    extras_require={
        "gpu": [
            "pytorch",
            "cupy",
        ],
    },
)