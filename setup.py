from setuptools import setup, find_packages

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(name='robust_hw',
      version='0.1.0',
      author='Nikolay Ivanov',
      author_email='nikolay.ivanov@desy.de',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/simply-nicky/robust_hw",
      paskages=find_packages(),
      package_dir={"": "robust_hw"},
      include_package_data=False,
      install_requires=['h5py', 'numpy', 'scipy'],
      classifiers=[
          "Programming Language :: Python",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent"
      ],
      python_requires='>=3.7')
