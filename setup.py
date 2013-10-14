import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "shifted_beta_geometric",
    version = "0.0.1",
    author = "JD Maturen",
    author_email = "jdmaturen@gmail.com",
    description = """An implementation of the shifted-beta-geometric (sBG) model from Fader and Hardie's "How to Project
            Customer Retention" (2006)""",
    license = "Apache 2",
    keywords = "clv crm customer retention data modeling",
    # url = "http://github.com/jdmaturen/shifted_beta_geometric_py",
    packages=['shifted_beta_geometric'],

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: Apache Software License",
    ],
)