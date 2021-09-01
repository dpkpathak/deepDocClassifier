import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepDocClassifier",
    version="0.0.1",
    author="Deepak Pathak",
    author_email="dpathak@uos.de",
    description="Reimplementation of DeepDocClassifier, https://ieeexplore.ieee.org/document/7333933",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=['torch',
                      'torchvision',
                      'pytorch-lightning',
                      'pytorch-lightning-bolts'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
