from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="callanalyser",
    version="0.1.0",
    author="Gulliver Handley",
    description="A Python-based system for analyzing Zoom call recordings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gulliverhandley/recorded_call_analyser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "callanalyser": ["prompts/*.txt"],
    },
    entry_points={
        "console_scripts": [
            "generate-timeline=callanalyser.examples.generate_timeline:main",
            "analyze-metrics=callanalyser.examples.analyze_metrics:main",
            "analyze-training=callanalyser.examples.analyze_training_metrics:main",
        ],
    },
) 