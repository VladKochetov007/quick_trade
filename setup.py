# -*- coding: utf-8 -*-
from distutils.core import setup
import setuptools

setup(
    name='quick_trade',
    author='Vlad Kochetov',
    author_email='vladyslavdrrragonkoch@gmail.com',
    packages=setuptools.find_packages(),
    version='2.1.11',
    description='Trading system for stocks, forex and others',
    long_description='Please, view page on github.',
    url='https://github.com/VladKochetov007/quick_trade',
    install_requires=[
        'iexfinance==0.4.3',
        'plotly==4.9.0',
        'ta==0.5.25',
        'scikit-learn==0.23.1',
        'tensorflow==2.2.0',
        'pykalman==0.9.5',
        'scipy==1.4.1',
        'tqdm==4.48.0',
        'numpy==1.18.5',
        'pandas==1.0.5',
    ],
    download_url='https://github.com/VladKochetov007/quick_trade/archive/2.1.11.tar.gz',
    keywords=['technical analysis', 'python3', 'trading'],
    license='cc-by-sa-4.0',
    classifiers=[
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3',
    ],
)
