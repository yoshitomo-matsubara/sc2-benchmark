from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

description = 'SC2 Benchmark: Supervised Compression for Split Computing.'
setup(
    name='sc2bench',
    version='0.1.0',
    author='Yoshitomo Matsubara',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yoshitomo-matsubara/sc2-benchmark',
    packages=find_packages(exclude=('configs', 'resources', 'script', 'tests')),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.1',
        'numpy',
        'pyyaml>=6.0.0',
        'scipy',
        'cython',
        'pycocotools>=2.0.2',
        'torchdistill>=1.0.0',
        'compressai>=1.2.3',
        'timm>=0.4.12'
    ],
    extras_require={
        'test': ['pytest'],
        'docs': ['sphinx', 'sphinx_rtd_theme']
    }
)
