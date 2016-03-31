from distutils.core import setup

setup(
    name='my-ml-data-loader',
    version='1.0',
    description='machine learning dataset loader for my experiment',
    author='masaponto',
    author_email='masaponto@gmail.com',
    url='masaponto.github.io',
    install_requires=['numpy', 'scikit-learn'],
    py_modules = ["my-ml-data-loader"],
    package_dir = {'': 'src'}
)
