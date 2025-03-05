# Sonara

Sonara is a Python project for audio processing and backend services. It includes functionalities such as audio recognition, segmentation, and more.

## Project Structure

```
├── sonara/
│   ├── __init__.py
│   └── backserver.py
├── tests/
│   ├── __init__.py
│   ├── test_basic.py
├── README.md
└── pyproject.toml
```
- **sonara/**: Contains the main code of the project.
- **tests/**: Contains the unit tests.
- **pyproject.toml**: Manages project dependencies and build configuration using Poetry.

## Installation

### Prerequisites

Ensure you have [Poetry](https://python-poetry.org/) installed:
```bash
pip install poetry
```

### Install Dependencies
To install the project dependencies (including the development dependencies), run:

```bash
poetry install --with dev
```

### Building the Project
To build the project package, run:
```bash
poetry build
```
After a successful build, the package files will be available in the dist/ directory.

### Running the Project
Depending on the module you want to run, you can use the following command. For example, to run the backend server:
```bash
poetry run backserver
```
Adjust the command if you want to run a different module.

### Running Tests

Run all tests:
```bash
poetry run pytest
```

Run a single test with log:
```bash
poetry run pytest -s --log-cli-level=INFO tests/test_llm_translate.py::test_translate_text
```
please make sure you enable logging in your test file:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Running Tests with Coverage
Run tests with coverage:
```bash
poetry run pytest --cov=sonara tests/
```

### Configuring Coverage (Optional)
If you want to customize the coverage configuration, create a .coveragerc file in the project root with the following content:
```ini
[run]
branch = True
source = sonara

[report]
show_missing = True
```
Then you can run:
```bash
poetry run coverage run -m pytest
poetry run coverage report -m
```

## Common Issues

- ModuleNotFoundError:

    If you encounter errors like `ModuleNotFoundError: No module named 'websockets'`, make sure to add the missing dependency. You can add it by running:

    ```bash
    poetry add websockets
    ```
    Then, reinstall dependencies:
    ```bash
    poetry install
    ```

## Contributing
Contributions are welcome! Please submit pull requests or open issues for bug reports and feature requests.