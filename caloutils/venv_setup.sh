#!/bin/bash
set -xe
python -m venv venv
source venv/bin/activate
pip install --upgrade pip flit ruff black mypy pytest pre-commit
pre-commit install

if [[ ${HOSTNAME} == max-*desy.de ]]; then
    source ${MODULESHOME}/init/bash
    module load maxwell gcc/9.3
fi

flit install --symlink
