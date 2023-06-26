[![Build Status](https://github.com/allen-cell-animated/abm-shape-collection/workflows/build/badge.svg)](https://github.com/allen-cell-animated/abm-shape-collection/actions?query=workflow%3Abuild)
[![Codecov](https://img.shields.io/codecov/c/gh/allen-cell-animated/abm-shape-collection?token=FMO7XIX5KH)](https://codecov.io/gh/allen-cell-animated/abm-shape-collection)
[![Lint Status](https://github.com/allen-cell-animated/abm-shape-collection/workflows/lint/badge.svg)](https://github.com/allen-cell-animated/abm-shape-collection/actions?query=workflow%3Alint)
[![Documentation](https://github.com/allen-cell-animated/abm-shape-collection/workflows/documentation/badge.svg)](https://allen-cell-animated.github.io/abm-shape-collection/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Collection of tasks for analyzing cell shapes.
Designed to be used both in [Prefect](https://docs.prefect.io/latest/) workflows and as modular, useful pieces of code.

# Installation

The collection can be installed using:

```bash
pip install abm-shape-collection
```

We recommend using [Poetry](https://python-poetry.org/) to manage and install dependencies.
To install into your Poetry project, use:

```bash
poetry add abm-shape-collection
```

# Usage

## Prefect workflows

All tasks in this collection are wrapped in a Prefect `@task` decorator, and can be used directly in a Prefect `@flow`.
Running tasks within a [Prefect](https://docs.prefect.io/latest/) flow enables you to take advantage of features such as automatically retrying failed tasks, monitoring workflow states, running tasks concurrently, deploying and scheduling flows, and more.

```python
from prefect import flow
from abm_shape_collection import <task_name>

@flow
def run_flow():
    <task_name>()

if __name__ == "__main__":
    run_flow()
```

See [cell-abm-pipeline](https://github.com/allen-cell-animated/cell-abm-pipeline) for examples of using tasks from different collections to build a pipeline for simulating and analyzing agent-based model data.

## Individual tasks

Not all use cases require a full workflow.
Tasks in this collection can be used without the Prefect `@task` decorator by simply importing directly from the module:

```python
from abm_shape_collection.<task_name> import <task_name>

def main():
    <task_name>()

if __name__ == "__main__":
    main()
```

or using the `.fn()` method:

```python
from abm_shape_collection import <task_name>

def main():
    <task_name>.fn()

if __name__ == "__main__":
    main()
```
