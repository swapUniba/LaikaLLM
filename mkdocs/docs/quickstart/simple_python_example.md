---
hide:
  - toc
---

# Simple Python example

Just like the [quickstart example foy .yaml](simple_yaml_example.md), in this simple experiment we will:

1. Use the `toys` [Amazon Dataset](../yaml_usage/available_implementations/available_datasets.md#amazondataset) and add 'item_' and 'user_' 
   prefixes to each item and user ids
2. Train the [GPT2Rec](../yaml_usage/available_implementations/available_models.md#gpt2rec) using the `distilgpt2` checkpoint on the 
   `SequentialSideInfoTask`
3. Evaluate results using `hit@10` and `hit@5` metrics

The model trained will be saved into *models/simple_experiment* path and the metrics results into
*reports/metrics/simple_experiment* path


```python title="Run the experiment"
from src.data.datasets.amazon_dataset import AmazonDataset
from src.data.tasks.tasks import SequentialSideInfoTask
from src.evaluate.evaluator import RecEvaluator
from src.evaluate.metrics.ranking_metrics import Hit
from src.model.models.gpt import GPT2Rec
from src.model.trainer import RecTrainer

if __name__ == "__main__":
    
    # data phase
    ds = AmazonDataset("toys", add_prefix_items_users=True)
    
    ds_splits = ds.get_hf_datasets()  # this returns a dict of hf datasets
    
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

As you can see, it's very **easy** to perform a complete experiment also via the *Python API*!


## Reducing dataset size for testing purposes

If you really want to  get a glimpse of the final results of the process without having to wait a lot, 
since *AmazonDataset* is quite big, you could *cut* the datasets size of each split for *testing purposes*!

This is the new *data phase* in which each dataset split size has been reduced

```python title="Data phase with reduced dataset size for testing"
from datasets import Dataset
from src.data.datasets.amazon_dataset import AmazonDataset

# data phase
ds = AmazonDataset("toys", add_prefix_items_users=True)

ds_splits = ds.get_hf_datasets()

train_split = Dataset.from_dict(ds_splits["train"][:100])
val_split = Dataset.from_dict(ds_splits["validation"][:100])
test_split = Dataset.from_dict(ds_splits["test"][:100])
```

As you can see, the **full integration** with state-of-the-art libraries makes *LaikaLLM feel like home*!
