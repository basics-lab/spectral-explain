# spectral-explain

### Dependencies and Installation 

1. Install Poetry using the official [instructions](https://python-poetry.org/docs/#installing-with-pipx). This package uses Poetry for dependency management and packaging. 

2. Clone the package
```
git clone git@github.com:basics-lab/spectral-explain.git
cd spectral-explain
```

Ensure you have the latest version of python 3.10 installed on your device. If you are using an arm Mac, you MUST install an arm native version of python 3.10. This is best done via the command: `homebrew python@3.10`. The universal installer on [python.org](https://www.python.org/downloads/release/python-31011/) may install the x86 version. To verify run the following:

```
source <path-of-installed-python-binary>/activate
python -c “import platform; print(platform.processor())”
```
and ensure that it says `arm`. 

3. Create a virtual Python environment and install the necessary dependencies
```
poetry install
```
This command installs the package `spectral_explain` in [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/).

there are likely to be a few errors. For anything that was not installed by poetry install it via pip

Major bug: https://github.com/python-poetry/poetry/issues/8623

### Running the Experiments

You can run the experiments with
```
poetry run python experiments/measure_r2.py
``` 
or 
```
poetry shell
python experiments/measure_r2.py
```

### Development

Since we install our package in editable mode, the changes made to the source code are reflected immediately without requiring reinstallation of the package `spectral_explain`.

To add a new dependency, run
```
poetry add <package-name>
```

For developing on VSCode or PyCharm, get the path of the Python interpreter with
```
poetry env info --executable
```
and use it as your interpreter.
