# -*- coding: utf-8 -*-
import os
from distutils.core import setup

import setuptools

setup(
    name='quick_trade',
    author='Vlad Kochetov',
    author_email='vladyslavdrrragonkoch@gmail.com',
    packages=setuptools.find_packages(),
    version='3.0.1',
    description='Trading system for stocks, forex and others',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    long_description_content_type='text/markdown',
    url='https://github.com/VladKochetov007/quick_trade',
    install_requires=[
        "iexfinance==0.4.3",
        "plotly==4.12.0",
        "ta==0.5.25",
        "scikit-learn==0.23.2",
        "tensorflow==2.3.1",
        "pykalman==0.9.5",
        "scipy==1.5.4",
        "numpy==1.19.2",
        "pandas==1.1.4",
        "python-binance==0.7.5",
        "requests==2.24.0",
    ],
    download_url='https://github.com/VladKochetov007/quick_trade/archive/3.0.tar.gz',
    keywords=['technical analysis',
              'python3',
              'trading',
              'binance',
              'trading-bot'
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
