Installation
============

pip
---
You can install `Fringes` directly from `Pypi <https://pypi.org/>`_ with ``pip``::

    pip install fringes

..
    ``poetry install`` does not work
    First, ensure that poetry is installed correctly as descibed on the [Poetry Website](https://python-poetry.org/docs/).\
    Secondly, ensure the correct python version is installed on your system, as specified in the file `pyproject.toml`\
    Third, this can be caused by a proxy which `pip` does not handle correctly.
    Manually setting the proxy in the Windows settings
    or even adding a system variable `https_proxy = http://YOUR_PROXY:PORT` can resolve this.

From Source
-----------
To get access to the very latest features and bugfixes you have three choices:

1. Clone `Fringes` from `GitHub <https://github.com/>`_ with ``git``::

    git clone https://github.com/comimag/fringes
    cd fringes

   Now you can install `Fringes` from the source wit ``pip``::

    pip install .

2. Directly install from GitHub repo with ``pip``::

    pip install git+git://github.com/comimag/fringes.git@main

   You can change ``main`` of the above command to the branch name or the
   commit you prefer.

3. You can simply place the `fringes` folder someplace importable, such as
   inside the root of another project. `Fringes` does not need to be "built" or
   compiled in any way.