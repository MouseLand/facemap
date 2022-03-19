import setuptools

install_deps = ['numpy>=1.16', 
                'scipy',
                'matplotlib',
                'natsort', 
                'tqdm', 
                'numba>=0.43.1',
                'opencv-python-headless', 
                'torch>=1.9',
                'torchvision',
                'umap-learn', 
                'pandas', 
                'scikit-image']

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="facemap",
    license="GPLv3",
    version="0.2.0",
    author="Carsen Stringer & Atika Syeda & Renee Tung",
    author_email="carsen.stringer@gmail.com",
    description="Processing motion SVDs of videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/FaceMap",
    packages=setuptools.find_packages(),
    install_requires = install_deps,
    tests_require = ['pytest', 'tqdm'],
    extras_require = {
        'gui': [
            'pyqtgraph==0.12.0',
            'pyqt5', 
            'pyqt5.sip',
        ]
    },
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
)
