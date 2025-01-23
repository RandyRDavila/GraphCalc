from setuptools import setup, find_packages

setup(
    name="graphcalc",
    version="0.1.13",
    author="Randy Davila",
    author_email="rrd6@rice.edu",
    description="A Python package for graph computation functions",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/randydavila/graphcalc",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",            # Numerical operations
        "networkx>=3.0",          # Graph-theoretic computations
        "pillow>=9.0",            # Image handling (if used for visualization)
        "PuLP>=2.0",              # Linear programming
        "matplotlib>=3.0",        # Plotting (if visualization is part of the package)
        "python-dateutil>=2.8",   # Date handling (if required by your package)
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    keywords="graph theory, networkx, graph computation",
    project_urls={
        "Documentation": "https://graphcalc.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/randydavila/graphcalc",
        "PyPI": "https://pypi.org/project/graphcalc/"
    },
)
