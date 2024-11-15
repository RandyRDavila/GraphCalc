Installation
============

This guide walks you through the installation process for GraphCalc, a Python package for calculating graph invariants using mixed-integer programming.

Prerequisites
-------------

1. **Python**: Ensure you have Python 3.7 or later installed. Check your Python version by running:

``python --version``

2. **Pip**: Make sure you have ``pip`` installed. You can check your ``pip`` version by running:

   ``pip --version``

3. **Virtual Environment (Optional)**: It is recommended to create a virtual environment for GraphcCalc to avoid conflicts with other Python packages. You can create a virtual environment using the following command:

   ``python -m venv .venv``

Activate the virtual environment by running:

   ``source .venv/bin/activate``

On Windows, you can activate the virtual environment by running:

   ``.venv\Scripts\activate``

To deactivate the virtual environment, run:

   ``deactivate``

For more information on virtual environments, refer to the `Python documentation <https://docs.python.org/3/library/venv.html>`__.

Installation
------------

You can install GraphCalc from the Python Package Index (PyPI) using ``pip``. Run the following command to install the latest version of GraphCalc:

   ``pip install graphcalc``

Verify the Installation
-----------------------

To confirm that GraphCalc was installed correctly, open a Python interpreter and try importing it:

   ```
   import graphcalc
   print(graphcalc.__version__)
   ```

Updating GraphCalc
-------------------

To update GraphCalc to the latest version, use:

   ``pip install --upgrade graphcalc``

Uninstalling GraphCalc
-----------------------

If you need to uninstall GraphCalc, run:

   ``pip uninstall graphcalc``

Troubleshooting
---------------

- **Compatibility Issues**: Ensure your Python version is 3.7 or later. Compatibility issues may arise with older Python versions.
- **Solver Installation**: If GraphCalc relies on specific solvers, refer to the package documentation or installation guide for instructions on installing compatible solvers.
