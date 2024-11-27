"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import setuptools

install_deps = [
    "numpy>=1.16,<2.0",
    "scipy",
    "natsort",
    "tqdm",
    "numba>=0.43.1",
    "opencv-python-headless<4.10",
    "torch>=1.9",
    "h5py",
    "scikit-learn",
]
docs_deps = [
    "sphinx>=3.0",
    "sphinxcontrib-apidoc",
    "sphinx_rtd_theme",
]
gui_deps = [
    "pyqtgraph>=0.12.0",
    "pyqt6",
    "pyqt6.sip",
    "qtpy",
    "matplotlib",
]

try:
    import torch

    a = torch.ones(2, 3)
    version0, version1 = torch.__version__.split(".")[:2]
    version0, version1 = int(version0), int(version1)
    if version1 >= 9 or version0 >= 2:
        install_deps.remove("torch>=1.9")
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="facemap",
    license="GPLv3",
    author="Carsen Stringer & Atika Syeda",
    author_email="carsen.stringer@gmail.com",
    description="Pose estimation and processing SVDs of videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mouseland/facemap",
    setup_requires=[
        "pytest-runner",
        "setuptools_scm",
    ],
    packages=setuptools.find_packages(),
    use_scm_version=True,
    install_requires=install_deps,
    tests_require=["pytest", "tqdm"],
    extras_require={
        "docs": docs_deps,
        "gui": gui_deps,
    },
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
    entry_points={"console_scripts": ["facemap = facemap.__main__:main"]},
)
