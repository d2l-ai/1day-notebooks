#!/bin/bash
set -e

if [ $# -eq 0 ]; then
	echo "Usage: bash $0 NOTEBOOKS"
	echo "E.g., bash run_notebooks.sh */*.ipynb"
	echo "Execute all the notebooks and save outputs (assuming with Python 3)."
	exit -1
fi

echo "Start to evaluate $@"

for f in $@; do
	echo "=== Executing $f"
	jupyter nbconvert --execute --ExecutePreprocessor.kernel_name=python3 --to notebook --ExecutePreprocessor.timeout=1200 --inplace $f
done
