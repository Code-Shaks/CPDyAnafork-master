from setuptools import setup, find_packages

setup(
    name='CPDyAna',
    version='0.1.0',
    packages=find_packages(),          # auto‑discover Python packages
    install_requires=[
        'numpy>=1.23.5',
        'matplotlib>=3.5.0',
        'scipy>=1.8.0',
        'pandas>=1.3.0'
    ],
    author='CNL Lab',
    author_email='abc1@gmail.com',
    description='Unified analysis & plotting tool for CPDyAna',
    url='https://github.com/mantha123/CPDyAnafork',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',

    # single console‑script entry point -> CPDyAna.py’s main()
    entry_points={
        "console_scripts": [
            "cpdyana=target.CPDyAna:main",
        ]
    },
)
