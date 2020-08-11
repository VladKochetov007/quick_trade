# -*- coding: utf-8 -*-import setuptools
from distutils.core import setup
import setuptools

setup(
    name='quick_trade',
    packages=setuptools.find_packages(),
    version='2.1.4',
    description='Trading system for stocks, forex and others',
    long_description=open('README_for_pypi.txt').read(),
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
    download_url='https://github.com/VladKochetov007/quick_trade/blob/master/dist/quick_trade-2.1.2.tar.gz',
    keywords=['technical analysis', 'python3', 'trading'],
    license='cc-by-sa-4.0',
    classifiers=[
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)