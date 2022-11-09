from setuptools import setup, find_packages

base_packages = [
    "scikit-learn>=1.0.0", 
    "pandas>=1.0.0", 
    "sentence-transformers>=2.2.2",
    "spacy>=3.4.2",
    "gensim>=4.2.0",
    "en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.4.1/en_core_web_lg-3.4.1-py3-none-any.whl"
    ]

test_packages = [
    "pytest>=4.0.2",
    "black>=19.3b0"
]

setup(
    name="articlevectorizer",
    version="0.1.0",
    author="Krum Arnaudov",
    packages=find_packages(),
    description="A scikit-learn API package to create embeddings.",
    license_files = ("LICENSE"),
    install_requires=base_packages,
)