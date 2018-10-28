import os
import sys
from setuptools import setup, find_packages


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

setup(
    name='transptrb',
    version='0.1dev',
    url='https://github.com/mwong009/TRANSPTRB19',
    author='Melvin Wong',
    author_email='melvin.wong@ryerson.ca',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    licence='GPLv3'
)
