from distutils.core import setup

setup(
    name='ml_utils',
    version='2.0',
    description='machine learning experiment utilities for my experiment',
    author='masaponto',
    author_email='masaponto@gmail.com',
    url='masaponto.github.io',
    install_requires=['numpy', 'scikit-learn'],
    py_modules=["dataset_loader", "dexter.dexter_parser", "ml_utils"],
    package_dir={'': 'src'}
)
