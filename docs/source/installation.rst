Installation
------------

TemporalVAE requires Python 3.9 or later. We recommend to use Miniconda_.

PyPI
^^^^

Install scVelo from PyPI_ using::

    pip install -U TemporalVAE

``-U`` is short for ``--upgrade``.
If you get a ``Permission denied`` error, use ``pip install -U TemporalVAE --user`` instead.


Development Version
^^^^^^^^^^^^^^^^^^^

To work with the latest development version, install from GitHub_ using::

    pip install git+https://github.com/StatBiomed/TemporalVAE-release/@master

Dependencies
^^^^^^^^^^^^

- `anndata <https://anndata.readthedocs.io/>`_ - annotated data object.
- `scanpy <https://scanpy.readthedocs.io/>`_ - toolkit for single-cell analysis.
- `numpy <https://docs.scipy.org/>`_, `scipy <https://docs.scipy.org/>`_, `pandas <https://pandas.pydata.org/>`_, `scikit-learn <https://scikit-learn.org/>`_, `matplotlib <https://matplotlib.org/>`_.
- `pytorch <https://pytorch.org/>`_ - deep learning frameworks.



In this version, TemporalVAE can only run on GPU, we will provide CPU version as soon as possible. If you run into issues or have any new feature requirement, do not hesitate to approach us or raise a `GitHub issue`_ or contact us.

.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _PyPI: https://pypi.org/project/scvelo
.. _Github: https://github.com/StatBiomed/TemporalVAE-release
.. _`Github issue`: https://github.com/StatBiomed/TemporalVAE-release/issues/new/choose
