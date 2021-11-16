from setuptools import setup, find_packages

setup(
    name='biddinggame',
    version='0.1',
    author='bgsource',
    packages=find_packages(),
    install_requires=[
        'scipy',        
        'numpy',
        'pandas',
        'torch',
        'matplotlib',
        'Mesa'
    ]
)