"""{{ cookiecutter.package_name }} - {{ cookiecutter.package_description }}"""

from pathlib import Path

__version__ = '{{ cookiecutter.package_version }}'
__author__ = '{{ cookiecutter.author_name }} <{{ cookiecutter.author_email }}>'
__all__ = []

PROJECT_DIR = Path(__file__).parents[1]
DATA_DIR = PROJECT_DIR / "data"