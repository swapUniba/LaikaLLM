---
hide:
  - navigation
  - toc
---

<p align="center">
  <img src="https://github.com/Silleellie/LaikaLLM/assets/26851363/a43ed66b-f420-40e3-b261-9cfa690648fa" alt="drawing" width="50%"/>
</p>

# LaikaLLM

LaikaLLM is a software, for researchers, that helps in setting up a repeatable, reproducible, 
replicable protocol for **training** and **evaluating** <ins>multitask</ins> LLM for recommendation!

*Features*:

- Two different model family implemented at the moment of writing (***T5*** and ***GPT2***)
- Fully vectorized *Ranking* (`NDCG`, `MAP`, `HitRate`, ...) and *Error* (`RMSE`, `MAE`) metrics
- Fully integrated with **WandB** monitoring service
- Full use of *transformers* and *datasets* libraries
- Easy to use (via `.yaml` configuration or *Python api*)
- Fast (Intended to be used for *consumer gpus*)
- Fully modular and easily extensible!

The **goal** of LaikaLLM is to be the starting point, a *hub*, for all developers which want to evaluate the capability 
of LLM models in the **recommender system** domain with a *keen eye* on **devops** best practices!

Want a glimpse of LaikaLLM? This is an example configuration which runs the whole experiment pipeline, starting
from **data pre-processing**, to **evaluation**:

```yaml
exp_name: to_the_moon
device: cuda:0
random_seed: 42

data:
  AmazonDataset:
    dataset_name: toys

model:
  T5Rec:
    name_or_path: "google/flan-t5-base"
  n_epochs: 10
  train_batch_size: 32
  train_tasks:
    - SequentialSideInfoTask
    - RatingPredictionTask

eval:
  eval_batch_size: 16
  eval_tasks:
    SequentialSideInfoTask:
      - hit@1
      - hit@5
      - map@5
    RatingPredictionTask:
      - rmse
```

The whole pipeline can then be executed by simply invoking `python laikaLLM.py -c config.yml`!

## Motivation

The adoption of LLM in the recommender system domain is a *new* research area, thus it's **difficult** to 
find **pre-made** and **well-built** software designed *specifically* for LLM.

With *LaikaLLM* the idea is to fill that gap, or at least "start the conversation" about the importance
of developing *accountable* experiment pipelines

## Credits

A heartfelt "thank you" to [P5](https://github.com/jeykigung/P5) authors which, with their work, inspired the idea
of this repository and for making available a 
[preprocessed version](https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G/view?usp=sharing) of the 
[Amazon Dataset](https://huggingface.co/datasets/amazon_us_reviews) which in this project I've used as starting point 
for further manipulation.

> Yes, the cute logo is A.I. generated. So thank you DALL-E 3! 

Project Organization
------------
    ├── 📁 data                          <- Directory containing all data generated/used
    │   ├── 📁 processed                     <- The final, canonical data sets used for training/validation/evaluation
    │   └── 📁 raw                           <- The original, immutable data dump
    │
    ├── 📁 models                        <- Directory where trained and serialized models will be stored
    │
    ├── 📁 reports                       <- Where metrics will be stored after performing the evaluation phase
    │   └── 📁 metrics                          
    │
    ├── 📁 src                           <- Source code of the project
    │   ├── 📁 data                          <- All scripts related to datasets and tasks
    │   │   ├── 📁 datasets                  <- All datasets implemented
    │   │   ├── 📁 tasks                     <- All tasks implemented
    │   │   ├── 📄 abstract_dataset.py       <- The interface that all datasets should implement
    │   │   ├── 📄 abstract_task.py          <- The interface that all tasks should implement
    │   │   └── 📄 main.py                   <- Script used to perform the data phase when using LaikaLLM via .yaml
    │   │
    │   ├── 📁 evaluate                  <- Scripts to evaluate the trained models
    │   │   ├── 📁 metrics                   <- Scripts containing different metrics to evaluate the predictions generated
    │   │   ├── 📄 abstract_metric.py        <- The interface that all metrics should implement
    │   │   ├── 📄 evaluator.py              <- Script containing the Evaluator class used for performing the eval phase
    │   │   └── 📄 main.py                   <- Script used to perform the eval phase when using LaikaLLM via .yaml
    │   │
    │   ├── 📁 model                     <- Scripts to define and train models
    │   │   ├── 📁 models                    <- Scripts containing all the models implemented
    │   │   ├── 📄 abstract_model.py         <- The interface that all models should implement
    │   │   ├── 📄 main.py                   <- Script used to perform the eval phase when using LaikaLLM via .yaml
    │   │   └── 📄 trainer.py                <- Script containing the Trainer class used for performing the train phase
    │   │
    │   ├── 📄 __init__.py               <- Makes src a Python module
    │   ├── 📄 utils.py                  <- Contains utils function for the project
    │   └── 📄 yml_parse.py              <- Script responsible for coordinating the parsing of the .yaml file
    │
    ├── 📄 LICENSE                       <- MIT License
    ├── 📄 laikaLLM.py                   <- Script to invoke via command line to use LaikaLLM via .yaml
    ├── 📄 params.yml                    <- The example .yaml config for starting using LaikaLLM
    ├── 📄 README.md                     <- The top-level README for developers using this project
    └── 📄 requirements.txt              <- The requirements file for reproducing the environment (src package)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
