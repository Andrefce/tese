from setuptools import setup, find_packages

setup(
    name="cardiac-shape-gnn",
    version="0.1.0",
    description="LV Shape Reconstruction via GNN trained on a Statistical Shape Model",
    author="YOUR NAME",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "vtk>=9.2",
        "trimesh>=4.0",
        "nibabel>=5.0",
        "scikit-image>=0.21",
        "scikit-learn>=1.3",
        "torch>=2.0",
        "torch-geometric>=2.3",
        "matplotlib>=3.7",
        "tqdm>=4.65",
    ],
)
