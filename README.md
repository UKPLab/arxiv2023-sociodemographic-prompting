# Sociodemographic prompting for subjective NLP tasks

This repository includes the code for zero-shot prompting various subjective NLP tasks using sociodemographic information.

Further details can be found in our EACL 2024 publication [Sensitivity, Performance, Robustness: Deconstructing the Effect of Sociodemographic Prompting](https://aclanthology.org/2024.eacl-long.159/).

> **Abstract:** Annotators' sociodemographic backgrounds (i.e., the individual compositions of their gender, age, educational background, etc.) have a strong impact on their decisions when working on subjective NLP tasks, such as toxic language detection. Often, heterogeneous backgrounds result in high disagreements. To model this variation, recent work has explored sociodemographic prompting, a technique, which steers the output of prompt-based models towards answers that humans with specific sociodemographic profiles would give. However, the available NLP literature disagrees on the efficacy of this technique — it remains unclear for which tasks and scenarios it can help, and the role of the individual factors in sociodemographic prompting is still unexplored. We address this research gap by presenting the largest and most comprehensive study of sociodemographic prompting today. We use it to analyze its influence on model sensitivity, performance and robustness across seven datasets and six instruction-tuned model families. We show that sociodemographic information affects model predictions and can be beneficial for improving zero-shot learning in subjective NLP tasks.However, its outcomes largely vary for different model types, sizes, and datasets, and are subject to large variance with regards to prompt formulations. Most importantly, our results show that sociodemographic prompting should be used with care when used for data annotation or studying LLM alignment.


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

Please use the following citation from the [ACL Anthology](https://aclanthology.org/):

```
@inproceedings{beck-etal-2024-sensitivity,
    title = "Sensitivity, Performance, Robustness: Deconstructing the Effect of Sociodemographic Prompting",
    author = "Beck, Tilman  and
      Schuff, Hendrik  and
      Lauscher, Anne  and
      Gurevych, Iryna",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.159",
    pages = "2589--2615"
}
```
