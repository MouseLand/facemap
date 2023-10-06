Installation
===================================

Please see the Github readme for the latest installation `instructions`_

.. _instructions: https://github.com/MouseLand/facemap#readme

Common installation issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

- If you have ``pip`` issues, there might be some interaction between pre-installed dependencies and the required Facemap dependencies. Please upgrade pip as follows:
  ::

    python -m pip install --upgrade pip


- While running ``python -m facemap``, if you receive the error: ``No module named PyQt5.sip``, then try uninstalling and reinstalling pyqt5 as follows:
  ::

    pip uninstall pyqt5 pyqt5-tools
    pip install pyqt5 pyqt5-tools pyqt5.sip

- If you are on Yosemite Mac OS, PyQt doesn't work, and you won't be able to install Facemap. More recent versions of Mac OS are supported. The software has been heavily tested on Ubuntu 18.04, and less well tested on Windows 10 and Mac OS. Please post an issue if you have installation problems.

Dependencies
~~~~~~~~~~~~~~~~~~~

Facemap (python package) relies on these awesome packages:

- `pyqtgraph`_
- `pyqt6`_
- `numpy`_ (>=1.13.0)
- `scipy`_ 
- `opencv`_
- `numba`_
- `natsort`_
- `pytorch`_
- `matplotlib`_
- `tqdm`_
- `UMAP`_

.. _pyqtgraph: http://pyqtgraph.org/
.. _pyqt6: http://pyqt.sourceforge.net/Docs/PyQt6/
.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy.org/
.. _opencv: https://opencv.org/ 
.. _numba: http://numba.pydata.org/numba-doc/latest/user/5minguide.html
.. _natsort: https://natsort.readthedocs.io/en/master/
.. _pytorch: https://pytorch.org
.. _matplotlib: https://matplotlib.org
.. _tqdm: https://tqdm.github.io
.. _UMAP: https://umap-learn.readthedocs.io/en/latest/

MATLAB package installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The matlab version supports SVD processing only and does not include the orofacial tracker. The package can be downloaded/cloned from github (no install required). It works in Matlab 2014b and above. The Image Processing Toolbox is necessary to use the GUI. For GPU functionality, the Parallel Processing Toolbox is required. If you don't have the Parallel Processing Toolbox, uncheck the box next to "use GPU" in the GUI before processing. Note this version is no longer supported.
