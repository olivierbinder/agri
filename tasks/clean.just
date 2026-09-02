# run clean tasks
[group('clean')]
clean: clean-build clean-coverage clean-mlruns clean-outputs clean-pytest clean-python clean-ruff

# clean build folders
[group('clean')]
clean-build:
    rm -rf dist/
    rm -rf build/

# clean coverage files
[group('clean')]
clean-coverage:
    rm -rf .coverage*

# clean mlruns folder
[group('clean')]
clean-mlruns:
    rm -rf mlruns/
    rm -rf mlruns.db

# clean outputs folder
[group('clean')]
clean-outputs:
    rm -rf outputs/*

# clean pytest cache
[group('clean')]
clean-pytest:
    rm -rf .pytest_cache/

# clean python caches
[group('clean')]
clean-python:
    find . -type f -name '*.py[co]' -delete
    find . -type d -name __pycache__ -exec rm -r {} \+

# clean ruff cache
[group('clean')]
clean-ruff:
    rm -rf .ruff_cache/

# clean venv folder
[confirm]
[group('clean')]
clean-venv:
    rm -rf .venv/
