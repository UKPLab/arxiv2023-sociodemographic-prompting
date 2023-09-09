# Sociodemographic prompting for subjective NLP tasks

This repository includes the code for zero-shot prompting various subjective NLP tasks using sociodemographic information.

Further details can be found in our publication [Sociodemographic prompting for subjective NLP tasks](https://aclanthology.org/2023.starsem-1.43/).


> **Abstract:** The sociodemographic background has a strong impact on the decisions made by annotators for subjective NLP tasks, such as hate speech detection, often leading to high disagreements.
To model this variation, recent work has explored sociodemographic prompting, a technique, which steers the output of prompt-based models towards answers that humans with specific sociodemographic profiles would give. 
However, the available NLP literature disagrees on the efficacy of this technique — it remains unclear, for which tasks and scenarios it can help and evaluations are limited to specific tasks only. 
We address this research gap by presenting the largest and most comprehensive evaluation of sociodemographic prompting today. 
Concretely, we evaluate several prompt formulations across seven datasets and six instruction-tuned model families. 
We find that (1) while sociodemographic prompting can be beneficial for improving zero-shot learning in subjective NLP tasks, (2) it is unpredictable across different models and (3) subject to large variance with regards to prompt formulation.
Thus, it is not a reliable proxy for traditional data annotation with a sociodemographically heterogeneous group of annotators.
Instead, we propose (4) to use the technique for identifying ambiguous instances to inform annotation efforts.


## Information

Contact person: Tilman Beck, tilman.beck@tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to e-mail us or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Project structure

* `data/` -- container for the datasets
* `analysis` -- Python/Jupyter notebook scripts for sampling, analyzing data and results and visualization
* `configs` -- Hydra configuration files for the experiments
* `scripts/` -- folder for utility scripts for running the different experiments
* `sd_prompting.py` -- main script for running the prompting experiments

## Requirements

* Python3.6 or higher
* PyTorch 1.10.2 or higher
* CUDA

## Data

We make use of the following datasets: 

* Toxicity Detection
  * Diverse Perspectives provided by [Kumar et al. 2021](https://www.usenix.org/system/files/soups2021-kumar.pdf)
  * Jigsaw by [Kaggle](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)
* Stance Detection
  * SemEval 2016 Task 6 provided by [Mohammad et al. 2016](https://aclanthology.org/S16-1003/)
  * Global Warming Stance Detection [Luo et al. 2020](https://doi.org/10.18653/v1/2020.findings-emnlp.296)
* Hate Speech Detection
  * Gabe Hate Corpus provided by [Kennedy et al. 2022](https://doi.org/10.31234/osf.io/hqjxn)
  * Waseem provided by [Waseem 2016](https://doi.org/10.18653/v1/W16-5618)

## Setup

* Clone the repository
```
$ git clone https://github.com/UKPLab/arxiv2023-sociodemographic-prompting
$ cd arxiv2023-sociodemographic-prompting
```
* Create the environment and install dependencies

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
* Download the data from the respective sources (see above)


## Running the experiments

To employ sociodemographic prompting with *all* (i.e. "ratings") sociodemographic attributes,
run the following command:

```
$ python sd_prompting.py --task_name=stance_semeval2016t6 --input_dir=/path/to/input/ --output_dir=/path/to/output/ --input_file=input_file_name --model_name=google/flan-t5-small --format_id=0 --order_id=0 --sociodemographics=true --sociodemographic_attribute=gender --fewshot=false
```


### Parameter description
**(change this as needed!)**

The minimal set of configuration parameters which need to be set are the following. There are more options to be found in `configs/` and `sd_prompting.py`. 

* `--model_name`
  * the name of the model to use (e.g. `google/flan-t5-small`)
* `--cache_dir`
  * The cache dir of the transformers library
* `--task_name`
  * The name of the task to run (choose from `[toxicity_diverse_perspectives, toxicity_jigsaw, hatespeech_ghc, hatespeech_twitter, stance_semeval2016t6, stance_gwsd, sentiment_diaz]`)  
* `--format_id`
  * The format of the prompt (choose from `[0,1,2]`)
* `--input_dir`
  * The directory where the data files are stored 
* `--output_dir`
  * The directory to write the experiment outputs 
* `--input_file`
  * The name of the input file (e.g. `stance_semeval2016t6_filtered_sample_n1000_seed24`)


## Citing

Please use the following citation:

```
@article{beck-etal-2023-sociodemographic-prompting,
    title = "Sociodemographic prompting for subjective NLP tasks",
    author = "Beck, Tilman  and
      Schuff, Hendrik and Lauscher, Anne  and
      Gurevych, Iryna",
    year = "2023",
    journal = "CoRR"
    url = "url",
}
```