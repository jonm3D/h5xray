from setuptools import setup, find_packages

setup(
    name="h5xray",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "h5py",
        "pandas",
        "seaborn",
        "Pillow", 
        "s3fs",
        "requests",
        "icepyx",
        "scikit-learn",
        "plotly",
    ],
    extras_require={
        "optional": ["python-igraph", "kaleido"]
    },
    entry_points={
        'console_scripts': [
            'h5xray=h5xray.h5xray:main',
            'h5xray-interactive=h5xray.interactive:main',
        ],
    },
    description="HDF5 File Visualizer",
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jonm3d/h5xray",
    author="Jonathan Markel",
    author_email="jonathanmarkel@gmail.com",
    license="BSD-3-Clause",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Utilities"
    ],
    python_requires='>=3.7',  # Set the minimum Python version required for your package
)
