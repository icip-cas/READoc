#!/bin/bash

python src/evaluation/suite_e2e.py --gold_dir data/$SUBSET/markdown --pred_dir output/$SUBSET/$SYSTEM --result_json output/$SUBSET/$SYSTEM.json