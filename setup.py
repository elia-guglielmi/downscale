from setuptools import setup, find_packages

with open("README","r") as readme_file:
      long_description=readme_file.read()

setup(
      name='downscale',
      version='1.0',
      description='Python geographical data downdscale Utilities',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Elia Guglielmi',
      license="MIT",
      classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT Licence",
            "Programming Language :: Python 3"
      ],
      packages=find_packages(),
      install_requires=[
            'PyYAML',
            'numpy',
            'xarray',
            'pandas',
            'geopandas',
            'regionmask',
            'scikit-learn',
            'joblib',
            'xesmf'
            ],
      python_requires="~=3.5"
)