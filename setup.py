from setuptools import setup
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'Getting indicators based on smart money concepts or ICT'
LONG_DESCRIPTION = 'A package that allows users to get access to smart money concepts and ICT concepts in the form of indicators easily.'

# Setting up
setup(
    name="smartmoneyconcepts",
    version=VERSION,
    author="Joshua Attridge",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=["smartmoneyconcepts"],
    install_requires=["pandas", "numpy"],
    keywords=['python', 'smart money', 'ict', 'indicators', 'trading', 'forex', 'stocks', 'crypto', 'order blocks', 'liquidity'],
    url="https://github.com/joshyattridge/smartmoneyconcepts",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
