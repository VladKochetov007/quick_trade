# -*- coding: utf-8 -*-
from distutils.core import setup

from setuptools import find_packages

with open('README.md') as file:
    long_desc = file.read()

__version__ = "7.0.3"

setup(
    name='quick_trade',
    author="Vlad Kochetov",
    author_email='vladyslavdrrragonkoch@gmail.com',
    packages=find_packages(),
    version=__version__,
    description='Library for easy management and customization of algorithmic trading.',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    project_urls={
        'Documentation': 'https://vladkochetov007.github.io/quick_trade/#/',
        'Source': 'https://github.com/VladKochetov007/quick_trade',
        'Twitter': 'https://twitter.com/quick_trade_tw',
        'Bug Tracker': 'https://github.com/VladKochetov007/quick_trade/issues'
    },
    install_requires=[
        'numpy==1.21.4',
        'plotly==5.4.0',
        'pandas==1.3.4',
        'ta==0.8.0',
        'tqdm==4.62.3',
        'ccxt==1.62.31',
    ],
    download_url=f'https://github.com/VladKochetov007/quick_trade/archive/{__version__}.tar.gz',
    keywords=[
        'technical-analysis',
        'python3',
        'trading',
        'trading-bot',
        'trading',
        'binance-trading',
        'ccxt',
    ],
    license='cc-by-sa-4.0',
    classifiers=[
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.0',
)
