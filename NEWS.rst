History
=======

0.7.1 (2023-09-21)
------------------

* Fix deprecated NumPy type alias (np.float)

0.7 (2018-09-20)
----------------

* Python 3 support
* Clean up tests and more flake8 (line lengths)

0.6 (2016-12-05)
----------------

* Fix pip installation, clean up setup procedure, flake8 and add README
* PiecewisePolynomial1DFit updated to work with scipy 0.18.0
* Delaunay2DScatterFit now based on scipy.interpolate.griddata, which is
  orders of magnitude faster, more robust and smoother. Its default
  interpolation changed from 'nn' (natural neighbours - no longer available)
  to 'cubic'.
* Delaunay2DGridFit removed as there is no equivalent anymore

0.5.1 (2012-10-29)
------------------

* Use proper name for np.linalg.LinAlgError

0.5 (2011-09-26)
----------------

* Initial release of scikits.fitting
