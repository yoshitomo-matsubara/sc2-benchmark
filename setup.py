from setuptools import setup, find_packages

import sc2bench

with open('README.md', 'r') as f:
    long_description = f.read()

description = 'SC2: Supervised Compression for Split Computing.'
setup(
    name='sc2bench',
    version=sc2bench.__version__,
    author='Yoshitomo Matsubara',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yoshitomo-matsubara/sc2-benchmark',
    packages=find_packages(exclude=('configs', 'resources', 'script', 'tests')),
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.10.0',
        'torchvision>=0.11.1',
        'numpy',
        'pyyaml>=5.4.1',
        'scipy',
        'cython',
        'pycocotools>=2.0.2',
        'torchdistill>=0.2.7',
        'compressai>=1.1.8'
    ],
    extras_require={
        'test': ['pytest']
    }
)
