"""setup.py for lbs"""

from setuptools import setup

install_requires = [
    'torchtext>=0.2.3',
    'torchvision>=0.1.8',
    'numpy>=0.14',
    'matplotlib>=2.2.2',
    'absl-py>=0.1.13',
    'scikit-learn>=0.19.1',
    'pylint>=1.9.1',
    'yapf>=0.22.0',
    'pyyaml>=3.12',
    'ray>=0.4.0',
    'scipy>=1.1.0'
]

setup(name="lbs", author="RISE Lab", install_requires=install_requires)
