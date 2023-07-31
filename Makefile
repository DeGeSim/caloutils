.PHONY: clean test docs build release_local release_remote install build_venv

clean:
	rm -rf build/ dist/ .eggs/
	find . -name '__pycache__' -exec rm -r {} +
	rm -rf .coverage htmlcov

test:
	pytest

docs:
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

build: clean
	flit build

release_local: clean test build
	flit publish


release_remote:
	bump2version patch
	git push --tags

install:
	# flit install --editable
	pip install --no-deps -e .


build_venv:
	python -m venv venv
	source venv/bin/activate; pip install --upgrade pip flit ruff black mypy pytest pre-commit
	source venv/bin/activate; pre-commit install
