[![Build status](https://allen-cell-animated.github.io/abm-shape-collection/_badges/build.svg)](https://github.com/allen-cell-animated/abm-shape-collection/actions?query=workflow%3Abuild)
[![Lint status](https://allen-cell-animated.github.io/abm-shape-collection/_badges/lint.svg)](https://github.com/allen-cell-animated/abm-shape-collection/actions?query=workflow%3Alint)
[![Documentation](https://allen-cell-animated.github.io/abm-shape-collection/_badges/documentation.svg)](https://allen-cell-animated.github.io/abm-shape-collection/)
[![Coverage](https://allen-cell-animated.github.io/abm-shape-collection/_badges/coverage.svg)](https://allen-cell-animated.github.io/abm-shape-collection/_coverage/)
[![Code style](https://allen-cell-animated.github.io/abm-shape-collection/_badges/style.svg)](https://github.com/psf/black)
[![Version](https://allen-cell-animated.github.io/abm-shape-collection/_badges/version.svg)](https://pypi.org/project/abm-shape-collection/)
[![License](https://allen-cell-animated.github.io/abm-shape-collection/_badges/license.svg)](https://github.com/allen-cell-animated/abm-shape-collection/blob/main/LICENSE)

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
