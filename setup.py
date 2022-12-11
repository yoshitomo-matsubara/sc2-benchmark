from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

description = 'SC2: Supervised Compression for Split Computing.'
setup(
    name='sc2bench',
    version='0.0.3-dev',
    author='Yoshitomo Matsubara',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yoshitomo-matsubara/sc2-benchmark',
    packages=find_packages(exclude=('configs', 'resources', 'script', 'tests')),
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.11.0',
        'torchvision>=0.12.0',
        'numpy',
        'pyyaml>=5.4.1',
        'scipy',
        'cython',
        'pycocotools>=2.0.2',
        'torchdistill>=0.2.7',
        'compressai>=1.1.8',
        'timm>=0.4.12'
    ],
    extras_require={
        'test': ['pytest']
    }
)
