from setuptools import setup, find_packages
import os

# Utility function to read the README file with specified encoding
def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()

# Function to read the requirements file
def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Parse the requirements from requirements.txt
requirements = parse_requirements("requirements.txt")

setup(
    name="dl_utils",
    version="0.1.0",
    author="Yichen Guo, Joshua Agar",
    author_email="yig319@lehigh.edu, jca92@drexel.edu",
    license="MIT",
    keywords="symmetry, classification",

    # Ensure correct package discovery for multiple packages in src/
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    install_requires=requirements,
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
)
