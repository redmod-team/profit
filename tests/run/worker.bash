#!/bin/bash
# profit test script
# testint profit.run.worker with mockup

# ensure interface is ready
if [[ ! -e ./interface.npy ]]; then
    ./memmap.py
fi

# set paths
export PROFIT_BASE_DIR=`pwd`
export PROFIT_CONFIG_PATH=`pwd`/profit.yaml

# run configuration
PROFIT_RUN_DIR="run_01"
export PROFIT_RUN_ID="01"

# empty run_dir
if [[ -e $PROFIT_RUN_DIR ]]; then
    rm -r $PROFIT_RUN_DIR
fi

# create run dir
mkdir -p $PROFIT_RUN_DIR
cd $PROFIT_RUN_DIR

profit-worker

cd ..

# PART II
# run configuration
PROFIT_RUN_DIR="run_02_1"
export PROFIT_RUN_ID="02"
export PROFIT_ARRAY_ID=1

# create run dir
mkdir -p $PROFIT_RUN_DIR
cd $PROFIT_RUN_DIR

profit-worker

# clean run dir
cd ..
rm -r $PROFIT_RUN_DIR
