.PHONY: clean-build clean-pyc clean

# SERVER (ENABLE THESE WHEN THE SERVER IS READY)
# HOST=127.0.0.1
# TEST_PATH=./
# run-server: python manage.py runserver
# @echo "run-server - to start`< SERVER-NAME >` to run the server."

help:
	@echo "clean - remove all build, test, coverage and python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "install - install the package to the active python's site-packages"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -rf {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

install: clean
	python setup.py install
