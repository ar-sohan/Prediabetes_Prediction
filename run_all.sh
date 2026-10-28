#!/bin/bash
set -e
python -m src.step1_train                  # Table 1
python -m experiments.exp3_missingness     # no LLM needed
python -m experiments.exp4_jitai           # no LLM needed
python -m experiments.exp5_safety --agent
python -m experiments.exp1_extraction 60   # Table 2 (LLM)
python -m experiments.exp2_memory 20       # Table 3 (LLM)
