from distutils.core import setup
import setuptools

setup(
    name='quick_trade',
    version='2.1',
    packages=['quick_trade'],
    license='cc',
    description="trading system",
    url='https://github.com/VladKochetov007/quick_trade',
    # long_description=open('README_for_pypi.txt').read(),
    install_requires=[
        'iexfinance==0.4.3',
        'numpy==1.18.5',
        'pandas==1.0.5',
        'plotly==4.9.0',
        'pykalman==0.9.5',
        'scikit-learn==0.23.1',
        'scipy==1.4.1',
        'ta==0.5.25',
        'tensorflow==2.2.0',
        'tqdm==4.48.0'
    ]
)