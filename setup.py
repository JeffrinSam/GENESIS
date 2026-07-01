from setuptools import setup, find_packages

setup(
    name="genesis-robot",
    version="1.0.0",
    description="GENESIS: Video-Conditioned Robot Learning — agentic video generation, navigation, and manipulation",
    author="Jeffrin Sam",
    url="https://github.com/JeffrinSam/GENESIS",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    install_requires=[
        "anthropic>=0.21.0",
        "flask>=2.3.0",
        "werkzeug>=2.3.0",
        "requests>=2.28.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pillow>=9.0.0",
        "opencv-python>=4.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "generation": [
            "anthropic>=0.21.0",
        ],
        "navigation": [
            "timm>=0.9.0",
            "einops>=0.6.0",
            "diffusers>=0.21.0",
            "transformers>=4.30.0",
        ],
        "manipulation": [
            "transformers>=4.35.0",
            "peft>=0.6.0",
            "einops>=0.6.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
)
