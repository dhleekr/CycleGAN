from setuptools import setup, find_packages

setup(
    name='cyclegan',
    packages=[package for package in find_packages()],
    install_requires=[
        'torch',
        'torchvision',
        'pyyaml',
        'tensorboard',
        'Pillow',
        'numpy',
        'wandb',
    ],
    python_requires='>=3',
    version='0.1.0',
)