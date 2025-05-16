from setuptools import setup, find_packages

setup(
    name='LibACCPS',
    version='1.0.0',
    description='Library for modeling and analyzing discrete automata, observers, and supervisor optimization',
    author='Enrico Mattana; Andrea Matticola',
    author_email='youremail@example.com',
    url='https://github.com/yourusername/LibACCPS',
    packages=find_packages(),
    install_requires=[
        'networkx',
        'graphviz'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)