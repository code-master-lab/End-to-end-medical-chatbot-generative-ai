# ------------------------------------------------------------
# STEP 44: Project packaging configuration (setup.py)
# ------------------------------------------------------------
# This file tells Python:
# - What this project is
# - How it should be packaged
# - How it can be installed and reused
#
# This is REQUIRED when:
# - You want to structure your project professionally
# - You want to import your own modules cleanly
# - You want to deploy or share the project

from setuptools import find_packages, setup


# ------------------------------------------------------------
# setup() defines the metadata and configuration of the project
# ------------------------------------------------------------
setup(

    # name:
    # - The name of your project/package
    # - Used when installing or referencing the project
    #
    # NOTE:
    # This is NOT the GitHub repo name
    # This is the Python package identity

    name='Generative AI Project',


    # version:
    # - Current version of the project
    # - Important for upgrades and dependency management
    #
    # "0.0.0" usually means:
    # - Initial development stage
    # - Not yet a stable release

    version='0.0.0',


    # author:
    # - Name of the project author
    # - Informational metadata

    author='Bappy Ahmed',


    # author_email:
    # - Contact email of the author
    # - Used in package metadata

    author_email='entbappy73@gmail.com',


    # packages:
    # - Automatically finds all Python packages in the project
    # - A package = folder containing __init__.py
    #
    # This allows you to import your own modules like:
    # from src.helper import ...
    #
    # Without this:
    # - Python may fail to recognize your internal modules

    packages=find_packages(),


    # install_requires:
    # - List of external dependencies required by this project
    #
    # Empty here because:
    # - Dependencies are handled via requirements.txt
    # - Or added later for production packaging

    install_requires=[]
)

