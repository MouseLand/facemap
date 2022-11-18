import setuptools

install_deps = [
    "numpy>=1.16",
    "scipy",
    "natsort",
    "tqdm",
    "numba>=0.43.1",
    "opencv-python-headless",
    "torch>=1.9",
    "pandas",
    "scikit-learn",
    "tables==3.6.1",
]
docs_deps = [
    "sphinx>=3.0",
    "sphinxcontrib-apidoc",
    "sphinx_rtd_theme",
]
gui_deps = [
    "pyqtgraph>=0.12.0",
    "pyqt5",
    "pyqt5.sip",
    "matplotlib",
    "umap-learn",
]

try:
    import torch

    a = torch.ones(2, 3)
    version = int(torch.__version__.split(".")[1])
    if version >= 9:
        install_deps.remove("torch>=1.9")
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="facemap",
    license="GPLv3",
    version="0.2.0",
    author="Carsen Stringer & Atika Syeda & Renee Tung",
    author_email="carsen.stringer@gmail.com",
    description="Pose estimation and processing SVDs of videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mouseland/facemap",
    packages=setuptools.find_packages(),
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
)
