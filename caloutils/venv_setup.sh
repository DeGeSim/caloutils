#!/bin/bash
set -xe
python -m venv venv
source venv/bin/activate

if [[ ${HOSTNAME} == max-*desy.de ]]; then
    source ${MODULESHOME}/init/bash
    module load maxwell gcc/9.3
fi

flit install --symlink
pre-commit install
