from setuptools import setup, find_packages

setup(
    name='h5xray',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'h5xray=h5xray.your_script_name:main',
        ],
    },
)

