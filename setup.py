import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "Algopylib",
    version = "0.0.3",
    author = "Anisha Agrawal, Dhaval Kumar and Shivam Saxena",
    author_email = "iiitalgopy@googlegroups.com",
    description = "A basic python library that has modules for maths and algorithms ",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/still-n0thing/Algopylib",
    project_urls = {
        "Bug Tracker" : "https://github.com/still-n0thing/Algopylib/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where = "src"),
    python_requires = ">=3.7",
)
