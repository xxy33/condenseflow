from setuptools import setup, find_packages

setup(
    name="condenseflow",
    version="0.1.0",
    description="CondenseFlow: Efficient Latent Space Collaboration for Multi-Agent LLM Systems",
    author="CondenseFlow Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "datasets>=2.18.0",
        "peft>=0.10.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
    ],
)
