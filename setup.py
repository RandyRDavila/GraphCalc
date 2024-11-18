from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="graphcalc",
    version="0.1.11",
    author="Randy Davila",
    author_email="rrd6@rice.edu",
    description="A Python package for graph computation functions",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/randydavila/graphcalc",
    packages=find_packages(),
    install_requires=requirements,  # Use requirements from the file
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    keywords="graph theory, networkx, graph computation",
    project_urls={
        "Documentation": "https://graphcalc.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/yourusername/graphcalc",
        "PyPI": "https://pypi.org/project/graphcalc/"
    },
)
