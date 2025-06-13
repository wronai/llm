#!/usr/bin/env python3
"""
WronAI Package Setup
Polish Language Model Training and Inference
"""

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="wronai",
    version="0.1.0",
    author=Tom Sapletta,
    author_email="wronai@softreck.dev",
    description="Polski model językowy - demokratyzacja AI dla języka polskiego",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wronai/llm",
    project_urls={
        "Bug Tracker": "https://github.com/wronai/llm/issues",
        "Documentation": "https://github.com/wronai/llm/docs",
        "Source Code": "https://github.com/wronai/llm",
    },
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "inference": [
            "fastapi>=0.103.0",
            "uvicorn>=0.23.0",
            "streamlit>=1.26.0",
        ],
        "advanced": [
            "flash-attn>=2.3.0",
            "deepspeed>=0.10.0",
            "ray>=2.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "wronai-train=scripts.train:main",
            "wronai-inference=scripts.inference:main",
            "wronai-prepare-data=scripts.prepare_data:main",
            "wronai-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "wronai": [
            "configs/*.yaml",
            "data/polish_stopwords.txt",
            "templates/*.html",
        ],
    },
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "natural language processing",
        "polish language",
        "large language model",
        "machine learning",
        "deep learning",
        "transformers",
        "nlp",
        "ai",
        "llm"
    ],
)