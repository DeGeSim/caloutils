#!/bin/bash
set -xe
python -m venv venv
source venv/bin/activate
pip install --upgrade pip flit
if [[ ${HOSTNAME} == max-*desy.de ]]; then
    source ${MODULESHOME}/init/bash
    module load maxwell gcc/9.3
fi

flit install --symlink
pre-commit install
echo "Run `source venv/bin/activate` to activate the enviroment"
