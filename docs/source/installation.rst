Installation
============

This guide walks you through the installation process for GraphCalc, a Python package for calculating graph invariants using mixed-integer programming.

Prerequisites
-------------

1. **Python**: Ensure you have Python 3.7 or later installed. Check your Python version by running:

.. code-block:: bash

    python --version

2. **Pip**: Make sure you have ``pip`` installed. You can check your ``pip`` version by running:

.. code-block:: bash

    pip --version

3. **Virtual Environment (Optional)**: It is recommended to create a virtual environment for GraphcCalc to avoid conflicts with other Python packages. You can create a virtual environment using the following command:

.. code-block:: bash

     python -m venv .venv

Activate the virtual environment by running

.. code-block:: bash

    source .venv/bin/activate

On Windows, you can activate the virtual environment by running

.. code-block:: bash

    .venv\Scripts\activate

To deactivate the virtual environment, run:

.. code-block:: bash

    deactivate

For more information on virtual environments, refer to the `Python documentation <https://docs.python.org/3/library/venv.html>`__.

Solvers
--------

**Linear and Integer Programming Solvers**: Some features in GraphCalc depend on third-party solvers.

   At least one of the following is required if you intend to use solver-based functions (e.g., `gc.solve_independent_set`):

    - **CBC** (recommended):

     .. code-block:: bash

        brew install cbc      # macOS
        sudo apt install coinor-cbc  # Debian/Ubuntu

   - **HiGHS** (alternative):

     .. code-block:: bash

        pip install highspy

    Or, for CLI support (e.g., from source or via Homebrew on macOS):

     .. code-block:: bash

        brew install highs  # macOS
        sudo apt install highs  # Debian/Ubuntu



   GraphCalc will attempt to automatically detect the solver if it is installed. You can also manually specify the solver in API calls.


Installation
------------

You can install GraphCalc from the Python Package Index (PyPI) using ``pip``. Run the following command to install the latest version of GraphCalc:

.. code-block:: bash

    pip install graphcalc

Verify the Installation
-----------------------

To confirm that GraphCalc was installed correctly, open a Python interpreter and try importing it:

.. code-block:: python

   import graphcalc
   print(graphcalc.__version__)


Updating GraphCalc
-------------------

To update GraphCalc to the latest version, use:

.. code-block:: bash

   pip install --upgrade graphcalc

Uninstalling GraphCalc
-----------------------

If you need to uninstall GraphCalc, run:

.. code-block:: bash

    pip uninstall graphcalc

Troubleshooting
---------------

- **Compatibility Issues**: Ensure your Python version is 3.7 or later. Compatibility issues may arise with older Python versions.
- **Solver Installation**: If GraphCalc relies on specific solvers, refer to the package documentation or installation guide for instructions on installing compatible solvers.
