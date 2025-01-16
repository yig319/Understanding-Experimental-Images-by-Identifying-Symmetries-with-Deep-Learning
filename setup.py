from setuptools import setup, find_packages

# Function to read the requirements file
def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Parse the requirements from requirements.txt
requirements = parse_requirements("requirements.txt")

setup(
    name="dl_utils",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,  # Use the parsed requirements here
    python_requires=">=3.7",
)