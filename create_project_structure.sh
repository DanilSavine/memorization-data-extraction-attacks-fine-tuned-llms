#!/bin/bash

# Create main directories
mkdir -p {data/{raw,processed,canaries},models/{gpt2,llama2_8b,meditron},src/{data_processing,models,metrics,utils},experiments,results,notebooks}

# Create data files
touch data/raw/PHEE_dataset.csv
touch data/processed/{train.json,validation.json}
touch data/canaries/{identical_canary.txt,perturbed_canary.txt}

# Create source files
touch src/data_processing/{__init__.py,preprocess.py,canary_insertion.py}
touch src/models/{__init__.py,fine_tuning.py}
touch src/metrics/{__init__.py,perplexity.py,exposure.py,mia.py,generation_accuracy.py}
touch src/utils/{__init__.py,helpers.py}

# Create experiment files
touch experiments/{experiment_config.yaml,run_experiment.py}

# Create results file
touch results/experiment_results.csv

# Create notebook
touch notebooks/analysis.ipynb

# Create root files
touch {requirements.txt,README.md}

echo "Project structure created successfully!"