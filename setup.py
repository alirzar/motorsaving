from setuptools import setup, find_packages

setup(
    name='saveman',
    version='1.0',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*",
                                    "tests"]),
    license='MIT',
    author='AliRezaei',
    url='*',
    install_requires=[],
    tests_require=[
        'pytest',
        'pytest-cov'
    ],
    setup_requires=['pytest-runner']
)
