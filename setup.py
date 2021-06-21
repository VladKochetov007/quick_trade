# -*- coding: utf-8 -*-
from distutils.core import setup

import setuptools
from quick_trade.utils import __version__, __author__

with open('./README.md') as file:
    long_desc = file.read()
with open('./requirements.txt', 'r') as file:
    install_requires = file.read().split('\n')

setup(
    name='quick_trade',
    author=__author__,
    author_email='vladyslavdrrragonkoch@gmail.com',
    packages=setuptools.find_packages(),
    version=__version__,
    description='Library for easy management and customization of algorithmic trading.',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url='https://github.com/VladKochetov007/quick_trade',
    install_requires=install_requires,
    download_url=f'https://github.com/VladKochetov007/quick_trade/archive/{__version__}.tar.gz',
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
