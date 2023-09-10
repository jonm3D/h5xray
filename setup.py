from setuptools import setup, find_packages

setup(
    name="h5xray",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "pandas",
        "h5py",
        "argparse",
    ],
    entry_points={
        'console_scripts': [
            'h5xray=h5xray:main',
        ],
    },
)
