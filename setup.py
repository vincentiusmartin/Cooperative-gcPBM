# This exists to enable import of functionality in the regression portion of the project
from setuptools import setup

# run: python3 setup.py develop

setup(
   name='coopgcpbm',
   version='1.0',
   description='TF Cooperativity probe generation and analyses',
   author='Farica Zhuang',
   author_email='farica.zhuang@duke.com',
   packages=['coopgcpbm'],
   python_requires='>=3.9'
)
