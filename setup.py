from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="graphcomp",
    version="0.1.0",
    author="Randy Davila",
    author_email="rrd6@rice.edu",
    description="A Python package for graph computation functions",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/randydavila/graphcomp",
    packages=find_packages(),
    install_requires=requirements,  # Use requirements from the file
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    keywords="graph theory, networkx, graph computation",
)
