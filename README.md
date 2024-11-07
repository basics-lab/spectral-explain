# spectral-explain

### Dependencies and Installation 

1. Install Poetry using the official [instructions](https://python-poetry.org/docs/#installing-with-pipx). This package uses Poetry for dependency management and packaging. 

2. Clone the package
```
git clone git@github.com:basics-lab/spectral-explain.git
cd spectral-explain
```

3. Install a virtual Python environment with necessary dependencies
```
poetry install
```
This command installs the package `spectral_explain` in [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/).

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