#!/bin/bash -l

echo "Hello world!"

module load conda/2021-09-22
python -c 'print(1 + 1)'
