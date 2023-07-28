clean:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

test: ## run tests quickly with the default Python
	pytest

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/caloutils.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ caloutils
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

build: clean
	flit build

release: clean test build
	flit publish


install: build
	flit install

venv: venv
	python -m venv venv
	source venv/bin/activate; pip install --upgrade pip flit ruff black mypy pytest pre-commit
	source venv/bin/activate; pre-commit install
