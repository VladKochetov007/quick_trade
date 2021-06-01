# -*- coding: utf-8 -*-
from distutils.core import setup

import setuptools

with open('./README.md') as file:
    long_desc = file.read()

setup(
    name='quick_trade',
    author='Vlad Kochetov',
    author_email='vladyslavdrrragonkoch@gmail.com',
    packages=setuptools.find_packages(),
    version='4.4',
    description='Trading system for crypto, forex, stocks and others',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url='https://github.com/VladKochetov007/quick_trade',
    install_requires=[
        'plotly==4.14.3'
        'pykalman==0.9.5'
        'numpy==1.20.3'
        'pandas==1.2.4'
        'ta==0.7.0'
        'scipy==1.6.3'
        'ccxt==1.50.69'
    ],
    download_url='https://github.com/VladKochetov007/quick_trade/archive/4.4.tar.gz',
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
