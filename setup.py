from setuptools import setup, find_packages

setup(
  name = 'xops',
  packages = find_packages(exclude=['examples']),
  version = '0.0.0',
  license='MIT',
  description = 'X-Transformers - Pytorch',
  author = 'Noah Kay',
  author_email = 'noahkay13@gmail.com',
  url = 'https://github.com/Frikallo/x-ops',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'tensor operations',
    'einstein notation',
    'fast operations',
  ],
  install_requires=[
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)