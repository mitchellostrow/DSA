import setuptools

setuptools.setup(
    name="DSA",
    version="0.0.1",
    url="https://github.com/mitchellostrow/DSA",

    author="Mitchell Ostrow",
    author_email="ostrow@mit.edu",

    description="Dynamical Similarity Analysis",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.9.3',
        'torch>=1.13.0'
        'tqdm>=4.32.2'
    ],
    extras_require={
        'dev': [
            'pytest>=3.7'
        ]
    },
)