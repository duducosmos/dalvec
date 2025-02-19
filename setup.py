#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
import os
from setuptools import setup, find_packages


def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding="utf-8") as f:
        return f.read()


setup(
    name="dalvec",
    license="Apache License 2.0",
    version='0.0.1',
    author='Eduardo S. Pereira, Matteo Costa',
    author_email='pereira.somoza@gmail.com',
    packages=find_packages("src"),
    package_dir={"": "src"},
    description="Vector model for PYDAL",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    install_requires=["pydantic",
                      "numpy",
                      "transformers",
                      "orjson == 3.10.6",
                      "langchain",
                      "pydal",
                      "sentence_transformers",
                      "scikit-learn"
                      ],
)
