# Simple Yaml example

In this simple experiment, we will:
1. Use the `toys` [Amazon Dataset](../yaml_usage/data_section.md#amazondataset) and add `item_` and `user_` 
   prefixes to each item and user ids
2. Train the [GPT2Rec](../yaml_usage/model_section.md#gpt2rec) using the `distilgpt2` checkpoint on the 
   `SequentialSideInfoTask`
3. Evaluate results using `hit@10` and `hit@5` metrics


!!! info
    Please remember that to invoke *LaikaLLM* using the `.yaml` configuration, the working directory should be
    the ***repository root***!


## Yaml config

Define your custom `params.yml:`

```yaml title="params.yml"
exp_name: simple_exp
device: cuda:0
random_seed: 42
log_wandb: true

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

## Invoke LaikaLLM

After defining the above `params.yml`, simply execute the experiment with 

```commandline title="Run the experiment"
python laikaLLM.py -c params.yml
```

> The model trained and the evaluation results will be saved into `models` and `reports/metrics`