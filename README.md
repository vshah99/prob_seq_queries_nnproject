# Predictive Querying for Neural Autoregressive Sequence Models


## Setup

Our experiments make use of 5 datasets. Links for downloading these files can be found below. After downloading, place the files in the `/data` folder. MOOCs and Shakespeare are included in the supplemental materials as a reference.
- [MOOCs](https://snap.stanford.edu/jodie/#datasets)
- [Mobile Apps](https://ucidatalab.github.io/uci-digital-evidence/datasets/#mobile-app-usage)
- [Amazon Reviews](https://nijianmo.github.io/amazon/)
- [Shakespeare](http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt)
- [Wikitext-v2](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1/test)

## Code Structure

To provide a general guide to the codebase, we have outlined the main folders/files and what they contain below.
- `seq_queries/` - General directory where all the core-functionality code is found. 
  + `samples.py` - Contains all query estimation methods, including sampling, search, and hybrid methods.
  + `tree.py` - Contains the functionality for the hybrid's query sequence path tree.
  + `experiments.py` - Orchestration functions. The primary function of interest is `sample_dynamic_target_token` which conducts hitting time query experiments.
  + `train.py` and `model.py` and `optim.py` - Model training and management.
  + `data.py` - Data preprocessing and loading.
  + `arguments.py` - Argparse command line interface.
- `scripts/` - Collection of scripts that leverage the core functionality (above) for specific tasks. The names are generally self-explanatory. Each of these is prepared to be run so long as the model and datasets can be found in the directory structure. Be sure to adjust paths accordingly. Below are a subset of the most important scripts that we provide.
  + `get_importance_sampling.py`
  + `get_uniform_mc_sampling.py`
  + `get_naive_sampling.py`
  + `get_beam_search_lower_bound.py`
  + `get_ground_truth.py`
  + `beam_search_ablation.py`
  + `entropy_ablation.py`
  + `temperature_ablation.py`
  + `train/*` - All training scripts
- `data/` and `models/` - Where data and models should be placed, with some data already present. The code assumes each dataset will be present inside of a folder with its corresponding name. The names we utilized are **apps**, **amazon**, **moocs**, **shakespeare**, and **wikitext**. Due to size constraints, we only include copies of the shakespeare and moocs datasets.

## Experimentation

All of our experimentation, including model training, query estimation, and all ablations, is completed through a single argparse CLI. There are two ways to conduct experiments in this setting. The first is to write a small script with `get_args()` included from the `arguments.py` file and then execute it, sending non-default arguments to the execution through the CLI directly. However, since there are many argparse arguments to leverage and this code is quite flexible, we also provide the ability to populate argparse arguments with a `.yaml` file. Providing a `.yaml` file will overwrite all other arguments, and an example can be found in `config/sample.yaml`. 

The paragraph above assumes that you wish to run a custom set of experiments with our software. If this is not the case, please refer to the many pre-made scripts in the `scripts/` folder for guidance and edit those as necessary. These can be run simply with `python scripts/{sample_script}.py` and is our recommended means of running this code.


