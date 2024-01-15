

from setuptools import setup

setup(
    name="SNLLS",
    packages= [
        "SNLLS",
        "SNLLS.model",
        "SNLLS.optimizers",
        "SNLLS.utils",
    ],
    install_requires=[
        "filelock",
        "Pillow",
        "torch",
        "fire",
        "humanize",
        "requests",
        "tqdm",
        "matplotlib",
        "scikit-image",
        "scipy",
        "numpy",
        "ipython",
        "accelerate",
        "einops",
        "scikit-learn",
        "h5py",
        "pandas",
        "open3d",
        "trimesh",
    ],
    author="ISCI Lab",
)



