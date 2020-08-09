from distutils.core import setup
import setuptools

setup(
    name='quick_trade',
    version='2.1.3',
    packages=setuptools.find_packages(),
    license='cc-by-sa-4.0',
    description="trading system",
    url='https://github.com/VladKochetov007/quick_trade',
    long_description=open('README_for_pypi.txt').read(),
    download_url='https://github.com/VladKochetov007/quick_trade/blob/master/dist/quick_trade-2.1.2.tar.gz',
    keywords=['trading', 'tensorflow', 'keras'],
    author='Vlad Kochetov',
    author_email='vladyslavdrrragonkoch@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
      ],
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
        'tqdm'
    ],
)
