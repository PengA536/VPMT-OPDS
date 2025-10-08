from setuptools import setup, find_packages

setup(
    name="volleyball-pose-tracking",
    version="1.0.0",
    author="Research Team",
    description="Volleyball player pose estimation and tracking",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.5",
        "opencv-python>=4.5.3",
        "tensorflow>=1.15.5",
        "torch>=1.9.0",
    ],
)
