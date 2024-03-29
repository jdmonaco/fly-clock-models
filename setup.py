"""Setup installation file for Flymodel"""

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='flymodel',
    version='0.0.1',
    description='Flymodel',
    long_description=long_description,
    url='https://github.com/jdmonaco/monacolab/tree/master/flymodel',
    author='Joseph Monaco',
    author_email='jmonaco@jhu.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='model simulation neuroscience',
    packages=['flymodel'])
