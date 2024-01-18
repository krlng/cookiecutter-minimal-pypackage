# fixed imports
import ast
import functools
import io
import os
from contextlib import redirect_stdout

import mlflow
import pandas as pd
from IPython import get_ipython
from IPython.core.magic import register_cell_magic


def decorator_log_mlflow(func, experiment_name, run_name):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            # Log parameters
            for i, arg in enumerate(args):
                mlflow.log_param(f"arg{i}", arg)
            for key, value in kwargs.items():
                mlflow.log_param(key, value)

            # Capture prints
            f = io.StringIO()
            with redirect_stdout(f):
                result = func(*args, **kwargs)
            out = f.getvalue()
            for line in out.split("\n"):
                if ":" in line:
                    key, value = [s.strip() for s in line.split(":")]
                    try:
                        value = float(value)
                        mlflow.log_metric(key, value)
                    except ValueError:
                        pass  # Ignore if value is not a number

            # Log output as an artifact
            if isinstance(result, pd.DataFrame):
                result.to_json("comparison.json")
                mlflow.log_artifact("comparison.json")
            elif result is not None:
                with open("output.txt", "w") as f:
                    f.write(str(result))
                mlflow.log_artifact("output.txt")

        return result

    return wrapper


def log_mlflow(_func=None, *, experiment_name="default", run_name="test"):
    def decorator(func):
        return decorator_log_mlflow(func, experiment_name, run_name)

    if _func is None:
        return decorator
    else:
        return decorator(_func)


@register_cell_magic
def mlflow_magic(line, cell):
    if " " in line:
        run_name, experiment_name = line.split(" ")
    else:
        run_name = "default"
        experiment_name = "default"

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    # Start an MLflow run
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        # Store the full cell content as a variable "code"
        mlflow.log_param("code", cell)

        # Execute the cell and capture the output
        _ns = get_ipython().user_ns
        # Capture prints
        f = io.StringIO()
        with redirect_stdout(f):
            result = exec(cell, _ns)
        out = f.getvalue()
        for line in out.split("\n"):
            if ":" in line:
                key, value = [s.strip() for s in line.split(":")]
                try:
                    value = float(value)
                    mlflow.log_metric(key, value)
                except ValueError:
                    pass  # Ignore if value is not a number

        # Function to extract variable names from the AST
        def extract_vars(node):
            vars = []
            if isinstance(node, ast.Name):
                vars.append(node.id)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    vars.extend(extract_vars(target))
            elif isinstance(node, ast.Expr):
                vars.extend(extract_vars(node.value))
            return vars

        # Parse the cell and extract variable names
        tree = ast.parse(cell)
        used_vars = set()
        for node in ast.walk(tree):
            used_vars.update(extract_vars(node))

        # Log only variables that were defined or used in the cell
        for var_name in used_vars:
            if var_name in _ns and isinstance(_ns[var_name], (int, float, str)):
                mlflow.log_param(var_name, _ns[var_name])

        # Store the output of the cell as an artifact
        for var_name, var_value in _ns.items():
            if isinstance(var_value, get_ipython().user_ns["pd"].DataFrame):
                # If it is a pandas dataframe use log_table for this
                var_value.to_csv(f"{var_name}.csv")
                mlflow.log_artifact(f"{var_name}.csv")
    return result
