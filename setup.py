#!/usr/bin/env python
from setuptools import setup

setup (
    name = 'gmind',
    version = '0.0.1',
    description = 'Deep learning to help find gravitational waves using PyCBC and Keras',
    author = 'Ligo Virgo Collaboration',
    author_email = 'alex.nitz@ligo.org',
    url = 'https://ligo-cbc.github.io',
    keywords = ['ligo', 'physics', 'gravity', 'deep learning', 'astronomy',
                'gravitational waves', 'machine learning'],
    install_requires = ['pycbc', 'keras', 'tensorflow'],
    scripts  = [
               ],
    packages = ['gmind'
               ],
)
