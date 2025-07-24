from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="graphcalc",
    version="1.0.5",
    author="Randy Davila",
    author_email="rrd6@rice.edu",
    description="A Python package for graph computation and invariant discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/randyrdavila/graphcalc",
    license="MIT",
    packages=find_packages(exclude=["tests", "docs"]),
    install_requires=[
        "numpy",
        "networkx",
        "pillow",
        "PuLP",
        "matplotlib",
        "python-dateutil",
        "pandas",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",  # Or "5 - Production/Stable"
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    keywords="graph theory, networkx, graph computation, pulp, graph invariants",
    project_urls={
        "Documentation": "https://graphcalc.readthedocs.io/en/latest/",
        "Source": "https://github.com/randyrdavila/graphcalc",
        "PyPI": "https://pypi.org/project/graphcalc/",
        "Bug Tracker": "https://github.com/randydavila/graphcalc/issues",
    },
)
