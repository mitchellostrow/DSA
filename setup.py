import setuptools

setuptools.setup(
    name="DSA",
    version="1.0.1",
    url="https://github.com/mitchellostrow/DSA",

    author="Mitchell Ostrow",
    author_email="ostrow@mit.edu",

    description="Dynamical Similarity Analysis",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'torch>=1.3.0',
        'kooplearn>=1.0.6',
        'pot>=0.9.3'
    ],
    extras_require={
        'dev': [
            'pytest>=3.7'
        ]
    },
)