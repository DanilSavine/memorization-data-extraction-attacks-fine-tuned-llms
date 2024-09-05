# Memorization in Fine-Tuned Large Language Models

This repository contains the code and resources for our study on memorization in fine-tuned large language models, with a focus on the medical domain. 
Research done while at the [PRAIRIE Institute](https://prairie-institute.fr/) at the PSL University in Paris, under the supervision of Prof. Olivier Cappé and Prof. Jamal Atif.

## Abstract

We investigate the mechanisms and factors influencing memorization in fine-tuned large language models (LLMs), using the PHEE dataset of pharmacovigilance events. Our research employs two main approaches: a membership inference attack to detect memorized data, and a generation task with prompted prefixes to assess verbatim reproduction. We analyze the impact of adapting different weight matrices in the transformer architecture, the relationship between perplexity and memorization, and the effect of increasing the rank in low-rank adaptation (LoRA) fine-tuning.

You can read the report here [here](docs/Memorization_in_Fine_Tuned_LLMs_SAVINE_2024.pdf)

## Key Findings

1. In the Attention Mechanism, fine-tuning Value and Output matrices contributes more significantly to memorization compared to Query and Key matrices.
2. Lower perplexity in the fine-tuned model correlates with increased memorization.
3. Higher LoRA ranks lead to increased memorization, but with diminishing returns at higher ranks.

## Repository Structure

```
llm-memorization-study/
├── data/
│   └── PHEE/  # PHEE dataset 
├── src/
│   ├── config.yaml  # Configuration file for model and dataset
│   ├── main_fine_tuning.py  # Scripts for LoRA fine-tuning
│   ├── model_setup.py  # Model setup and configuration
│   ├── main_membership_inference.py  # Membership inference attack implementation
│   ├── main_prompted_generation.py  # Generation with prompted prefix
│   └── utils.py  # Utility functions
├── notebooks/ # gitignored directory for Jupyter notebooks
│   ├── analysis.ipynb  # Jupyter notebook for result analysis
│   └── visualizations.ipynb  # Jupyter notebook for creating plots
├── results/ # gitignored directory for storing experimental results
│   ├── figures/  # Generated plots and figures
│   └── raw/  # Raw experimental results
├── requirements.txt
├── README.md
└── LICENSE
```

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/DanilSavine/LLM-finetuning-extraction
   cd llm-memorization-study
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Usage

0. Configure the model and dataset in `src/config.yaml`.

1. Fine-tune the model:
   ```
   python src/main_fine_tuning.py
   ```

2. Run membership inference attack:
   ```
   python src/main_membership_inference.py
   ```

3. Perform prompted generation:
   ```
   python src/main_prompted_generation.py
   ```

4. Analyze results and generate plots using the Jupyter notebooks in the `notebooks/` directory.

## Contributing

We welcome contributions to this project. Please feel free to submit issues, fork the repository and send pull requests!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Use of GitHub Copilot by Microsoft for code assistance
- Claude by Anthropic for report editing and formatting
- PRAIRIE Research Institute, PSL University for research support

## Citation

If you use this code or our findings in your research, please cite our work:

```
@misc{savine2024memorization,
  author = {Danil Savine, Jamal Atif, Olivier Cappé},
  title = {Memorization in Fine-Tuned Large Language Models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DanilSavine/LLM-finetuning-extraction}}
}
```
