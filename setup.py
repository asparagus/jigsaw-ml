import os
import setuptools


# WORKDIR is set by the Docker images, otherwise use current setup.
workdir = os.getcwd()

# Read from requirements.txt for consistency
requirements_path = os.path.join(workdir, 'requirements.txt')

try:
    with open(requirements_path) as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    raise RuntimeError(f'Could not find requirements file.')


setuptools.setup(
    name='jigsaw',
    version='0.1',
    description='Jigsaw ML package',
    author='Ariel Perez',
    author_email='arielperezch@gmail.com',
    url='https://github.com/asparagus/jigsaw-ml',
    install_requires=requirements,
    packages=['jigsaw'],
)
