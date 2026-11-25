# TrillionaireFX — one-command tasks. On Windows without `make`, use:  .\make.ps1 <target>
PY ?= .venv/bin/python
ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
endif

.PHONY: setup run test lint typecheck fetch status quality sources research validate daily check

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

run:
	$(PY) -m tfx run

fetch:
	$(PY) -m tfx fetch

status:
	$(PY) -m tfx status

sources:
	$(PY) -m tfx sources

quality:
	$(PY) -m tfx quality

research:
	$(PY) -m tfx research

validate:
	$(PY) -m tfx validate

daily:
	$(PY) -m tfx daily

test:
	$(PY) -m pytest --cov=tfx --cov-report=term-missing

lint:
	$(PY) -m ruff check tfx tests

typecheck:
	$(PY) -m mypy

check: lint typecheck test
