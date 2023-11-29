<p align="center">
  <img src="https://github.com/Silleellie/LaikaLLM/assets/26851363/a43ed66b-f420-40e3-b261-9cfa690648fa" alt="drawing" width="60%"/>
</p>

<p align="center">
  <a href="https://silleellie.github.io/LaikaLLM/"><img
    src="https://img.shields.io/badge/Documentation-1d1625?logo=readthedocs&style=for-the-badge&logoColor=white"
    alt="Documentation"
  /></a>
  <a href="https://huggingface.co/"><img
    src="https://tinyurl.com/2p9ft7xf"
    alt="Hugging Face"
  /></a>
  <a href="https://pytorch.org/"><img
    src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"
    alt="Pythorch"
  /></a>
  <a href="https://wandb.ai/"><img
    src="https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black"
    alt="WandB"
  /></a>
  <a href="https://hub.docker.com/r/silleellie/laikallm"><img
    src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white"
    alt="Docker"
  /></a>
</p>

# LaikaLLM
[[Documentation](https://silleellie.github.io/LaikaLLM/)][[Sample Experiments](sample_experiments)]

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

If you want to have a full view on how experiments are visualized in WandB and more, head up to [sample_experiments!](sample_experiments) 

## Motivation

The adoption of LLM in the recommender system domain is a *new* research area, thus it's **difficult** to 
find **pre-made** and **well-built** software designed *specifically* for LLM.

With *LaikaLLM* the idea is to fill that gap, or at least "start the conversation" about the importance
of developing *accountable* experiment pipelines

## Installation

### From source

*LaikaLLM* requires **Python 3.10** or later, and all packages needed are listed in 
[`requirements.txt`](requirements.txt)

- Torch with cuda **11.7** has been set as requirement for reproducibility purposes, but feel free to change the cuda
  version with the most appropriate for your use case!

To install **LaikaLLM**:

1. Clone this repository:
  ```
  git clone https://github.com/Silleellie/LaikaLLM.git
  ```
2. Install the requirements:
  ```
  pip install -r requirements.txt
  ```
3. Start experimenting!
  - Use LaikaLLM via *Python API* or via `.yaml` config!

**NOTE**: It is **highly** suggested to set the following environment variables to obtain *100%* reproducible results of
your experiments:

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
```

You can check useful info about the above environment variables [here](https://docs.python.org/3.3/using/cmdline.html#envvar-PYTHONHASHSEED) and [here](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)

### Via Docker Image

Simply pull the latest [LaikaLLM Docker Image](https://hub.docker.com/r/silleellie/laikallm) 
which includes every preliminary step to run the project, including setting `PYTHONHASHSEED` and
`CUBLAS_WORKSPACE_CONFIG` for reproducibility purposes

## Usage

*Note:* when using LaikaLLM, the working directory should be set to the root of the repository!

*LaikaLLM* can be used in two different ways:

- `.yaml` config
- *Python API*

Both use cases follow the **data-model-evaluate** logic, in *code* and *project* structure, but also in the effective
usage of LaikaLLM

In the documentation there are *extensive* examples for both use cases, what follows is a small example of the same
experiment using the `.yaml` config and the *Python API*.

In this simple experiment, we will:

1. Use the `toys` Amazon Dataset and add 'item' and 'user' prefixes to each item and user ids 
2. Train the **distilgpt2** model on the SequentialSideInfoTask
3. Evaluate results using `hit@10` and `hit@5`

### Yaml config

- Define your custom `params.yml:`

  ```yaml
  exp_name: simple_exp
  device: cuda:0
  random_seed: 42
  
  data:
    AmazonDataset:
      dataset_name: toys
      add_prefix_items_users: true
  
  model:
    GPT2Rec:
      name_or_path: "distilgpt2"
    n_epochs: 10
    train_batch_size: 8
    train_tasks:
      - SequentialSideInfoTask
  
  eval:
    eval_batch_size: 4
    eval_tasks:
      SequentialSideInfoTask:
        - hit@10
        - hit@5
  ```
- After defining the above `params.yml`, simply execute the experiment with `python laikaLLM.py -c params.yml`

  - The model trained and the evaluation results will be saved into `models` and `reports/metrics`

### Python API

```python
from src.data.datasets.amazon_dataset import AmazonDataset
from src.data.tasks.tasks import SequentialSideInfoTask
from src.evaluate.evaluator import RecEvaluator
from src.evaluate.metrics.ranking_metrics import Hit
from src.model.models.gpt import GPT2Rec
from src.model.trainer import RecTrainer

if __name__ == "__main__":
    
    # data phase
    ds = AmazonDataset("toys", add_prefix_items_users=True)
    
    ds_splits = ds.get_hf_datasets()
    
    train_split = ds_splits["train"]
    val_split = ds_splits["validation"]
    test_split = ds_splits["test"]
    
    # model phase
    model = GPT2Rec("distilgpt2",
                    training_tasks_str=["SequentialSideInfoTask"],
                    all_unique_labels=list(ds.all_items))
    
    trainer = RecTrainer(model,
                         n_epochs=10,
                         batch_size=8,
                         train_sampling_fn=ds.sample_train_sequence,
                         output_dir="models/simple_experiment")
    
    trainer.train(train_split)
    
    # eval phase
    evaluator = RecEvaluator(model, eval_batch_size=4)
    
    evaluator.evaluate_suite(test_split,
                             tasks_to_evaluate={SequentialSideInfoTask(): [Hit(k=10), Hit(k=5)]},
                             output_dir="reports/metrics/simple_experiment")
```

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
