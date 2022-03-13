# Installation (Python)

This package only supports python 3. We recommend installing python 3 with **[Anaconda](https://www.anaconda.com/download/)**.

#### For latest released version (from PyPI)

Run the following
~~~
pip install facemap
~~~
or the following for using GUI:
~~~~
pip install facemap[gui]
~~~~

To upgrade Facemap (package [here](https://pypi.org/project/facemap/)), within the environment run:
~~~~
pip install facemap --upgrade
~~~~

#### Using the environment.yml file (recommended)

1. Download the `environment.yml` file from the repository
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path
3. Run `conda env create -f environment.yml`
4. To activate this new environment, run `conda activate facemap`
5. You should see `(facemap)` on the left side of the terminal line. Now run `python -m facemap` and you're all set.

#### For latest development version 

To install updated github version using pip:
1. Run `pip install git+https://www.github.com/mouseland/facemap.git`

Alternatively,

1. Clone the github repository: `git clone https://www.github.com/mouseland/facemap.git`
2. Change directory to facemap folder `cd facemap`
3. Run `conda env create -f environment.yml`
4. To activate this new environment, run `conda activate facemap`
5. You should see `(facemap)` on the left side of the terminal line. Now run `python -m facemap` and you're all set.

### Common installation issues

If you have pip issues, there might be some interaction between pre-installed dependencies and the ones FaceMap needs. First thing to try is
~~~~
python -m pip install --upgrade pip
~~~~

While running `python -m facemap`, if you receive the error: `No module named PyQt5.sip`, then try uninstalling and reinstalling pyqt5
~~~
pip uninstall pyqt5 pyqt5-tools
pip install pyqt5 pyqt5-tools pyqt5.sip
~~~

If you are on Yosemite Mac OS, PyQt doesn't work, and you won't be able to install Facemap. More recent versions of Mac OS are fine.

The software has been heavily tested on Ubuntu 18.04, and less well tested on Windows 10 and Mac OS. Please post an issue if you have installation problems.

### Pyhton dependencies

Facemap python relies on these awesome packages:
- [pyqtgraph](http://pyqtgraph.org/)
- [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/)
- [numpy](http://www.numpy.org/) (>=1.13.0)
- [scipy](https://www.scipy.org/)
- [opencv](https://opencv.org/)
- [numba](http://numba.pydata.org/numba-doc/latest/user/5minguide.html)
- [natsort](https://natsort.readthedocs.io/en/master/)
- [PyTorch](https://pytorch.org)
- [Matplotlib](https://matplotlib.org)
- [SciPy](https://scipy.org)
- [tqdm](https://tqdm.github.io)
- [pandas](https://pandas.pydata.org)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)


# Installation (MATLAB) [SVD processing only]

The matlab version needs to be downloaded/cloned from github (no install required). It works in Matlab 2014b and above - please submit issues if it's not working. The Image Processing Toolbox is necessary to use the GUI. For GPU functionality, the Parallel Processing Toolbox is required. If you don't have the Parallel Processing Toolbox, uncheck the box next to "use GPU" in the GUI before processing.
