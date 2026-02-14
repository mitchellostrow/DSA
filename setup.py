import setuptools

setuptools.setup(
    name="dsa-metric",
    version="2.0.0",
    url="https://github.com/mitchellostrow/DSA",
    author="Mitchell Ostrow",
    author_email="ostrow@mit.edu",
    description="Dynamical Similarity Analysis",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.24.0,<2",
        "torch>=1.3.0",
        "pot",
        "omegaconf",
        "pydmd",
        "tqdm",
        "optht", #for havok in pykoopman
        "derivative", #for pykoopman
        "prettytable"
    ],
    extras_require={
        "dev": ["pytest>=3.7"],
        "umap": ["umap-learn"],
    },
)
