from setuptools import setup, find_packages

setup(
    name='LLMToolkit',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add any library dependencies here
        # For example: 'transformers>=4.0.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
