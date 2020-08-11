# -*- coding: utf-8 -*-import setuptools
from distutils.core import setup
import setuptools

setup(
    name='quick_trade',
    author='Vlad Kochetov',
    author_email='vladyslavdrrragonkoch@gmail.com',
    packages=setuptools.find_packages(),
    version='2.1.5',
    description='Trading system for stocks, forex and others',
    long_description='Please, view page on github.',
    url='https://github.com/VladKochetov007/quick_trade',
    install_requires=[
        'iexfinance',
        'numpy',
        'pandas',
        'plotly',
        'pykalman',
        'scikit-learn',
        'scipy',
        'ta',
        'tensorflow',
    ],
    download_url='https://github.com/VladKochetov007/quick_trade/raw/master/dist/quick_trade-2.1.5.tar.gz',
    keywords=['technical analysis', 'python3', 'trading'],
    license='cc-by-sa-4.0',
    classifiers=[
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)