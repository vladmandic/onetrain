#!/usr/bin/env bash

if [[ -f venv/bin/activate ]]
then
    source venv/bin/activate
else
    echo "Error: Cannot activate python venv"
    exit 1
fi
python onetrain.py $*
deactivate
