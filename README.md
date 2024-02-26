# PyMotifs:

## Structure of the repo:

The source package is located in `src`. Please refer to the `notebooks`
folder to see examples on how to use this package.

## Installation:

### Create a virtual environment

With conda: `conda create -n pymotifs python=3.11`

### To use the last stable version of the package

- Clone the `main` branch
- Activate your environment: `conda activate pymotifs`
- Install with `pip install .` at the source

### Install development branch

- Install with `pip install ."[dev]"` at the source
- Install pre-commit hooks with `pre-commit install`
- Start to code

### Notebooks

You must install the package in your ipython kernel. For example, you can
create a new kernel corresponding to your environement with: `install
ipython kernel install --name "pymotifs" --user`


### Notes

- Careful about **scipy** and **joblib** versions. There are issues with
  joblib when performing tasks in multiprocessing. The package was tested
  with **joblib==1.3.2** and **scipy==1.11.4**.
