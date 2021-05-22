# -*- coding: utf-8 -*-
import setuptools
from distutils.core import setup

with open('./README.md') as file:
    long_desc = file.read()

setup(
    name='quick_trade',
    author='Vlad Kochetov',
    author_email='vladyslavdrrragonkoch@gmail.com',
    packages=setuptools.find_packages(),
    version='4.2.1',
    description='Trading system for stocks, forex and others',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url='https://github.com/VladKochetov007/quick_trade',
    install_requires=[
        'iexfinance',
        'plotly',
        'pykalman',
        'numpy',
        'pandas',
        'python-binance==0.7.5',
        'ta==0.7.0',
        'scipy',
        'scikit-learn',
        'tensorflow==2.5.0',
        'ccxt',
    ],
    download_url='https://github.com/VladKochetov007/quick_trade/archive/4.2.tar.gz',
    keywords=[
        'technical-analysis',
        'python3',
        'trading',
        'binance',
        'trading-bot',
        'trading',
        'binance-trading',
        'ccxt'
    ],
    license='cc-by-sa-4.0',
    classifiers=[
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3',
    ],
)
