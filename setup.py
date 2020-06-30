from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='word_prism',
    version='0.0.1',
    description='word prism project',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
	#indlude_package_data=True,
	package_data={'': ['README.md']},
	install_requires=[
        'numpy', 'torch', 'hilbert-experiments'
    ]
)