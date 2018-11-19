import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="facemap",
    version="0.0.4",
    author="Carsen Stringer",
    author_email="carsen.stringer@gmail.com",
    description="Processing motion SVDs of videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/FaceMap",
    packages=setuptools.find_packages(),
    install_requires = ['ffmpeg','pyqtgraph', 'PyQt5', 'numpy>=1.13.0', 'scipy','matplotlib','pims'],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
)
