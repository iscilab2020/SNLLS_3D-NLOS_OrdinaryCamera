

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
        "einops",
        "scikit-learn",
        "h5py",
        "pandas",
        "open3d",
        "trimesh",
        "imageio",
        "ffmpeg"
    ],
    author="ISCI Lab",
)



