.PHONY: quality style test test-examples

# Check that source code meets quality standards

quality:
	black --check --line-length 119 --target-version py35 *.py
	isort --check-only *.py
	flake8 *.py

# Format source code automatically

style:
	black --line-length 119 --target-version py35 *.py
	isort *.py
