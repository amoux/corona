from setuptools import find_packages, setup
from Cython.Build import cythonize

PACKAGES = find_packages()

setup(
    name='corona_nlp',
    packages=PACKAGES,
    version='0.0.2',
    url='https://github.com/amoux/corona_nlp',
    author='Carlos A. Segura Diaz De Leon',
    author_email='carlosdeveloper2@gmail.com',
    license='MIT',
    zip_safe=False,
    ext_modules=cythonize("corona_nlp/dataset.pyx")
)
