# Model section

In the model section, you should define the model to use (and its parameters), along with the training parameters.
**Note**: The model to use must be the *first* attribute of the `model` section!

```yaml title="Model section"
model:
  MODEL_TO_USE:
    PARAM1: VAL1
    PARAM2:
      - VAL2
      - VAL3
    ...
  
  
  # Sequence of tasks to use during the training phase of the model.
  # Each sample of the train dataset will be applied to one of the followings or
  # all of them, depending on the `train_task_selection_strat` parameter
  #
  # Required
  train_tasks:
    - SequentialSideInfoTask
    - RatingPredictionTask
  
  # When training according to the multitask paradigm,
  # there are two different strategies available for choosing the task to apply
  # for the particular sample of the training set currently processed:
  # - "all" will apply, for the particular sample, ALL training tasks defined # (2)
  # - "random" will apply, for the particular sample, ONE training task among those defined choosen at random # (3)
  #
  # Optional, Default: "all"
  train_task_selection_strat: all
  
  # If set, the validation phase is performed at the end of each epoch of training:
  # - In this case, the best model will be saved according to the `monitor_metric` value
  # val_task should be set to one of the available tasks
  #
  # Optional, Default: null
  val_task: null
  
  # If `val_task` parameter is set, and thus the validation phase is performed,
  # you could specify the exact template of the `val_task` to use for validation.
  # If `val_task` is set but this parameter is set to null, then validation is performed
  # by choosing random templates of the `val_task` to apply to each sample of the
  # val dataset
  #
  # Optional, Default: null
  val_task_template_id: null
  
  # Number of epochs to perform during the training phase
  #
  # Optional, Default: 10
  n_epochs: 10
  
  # If `val_task` is set, and thus the validation phase is performed,
  # you can change the metric that should be used in order to save the best model
  #
  # Optional, Default: loss
  monitor_metric: loss
  
  # The batch size to use during the training phase
  #
  # Optional, Default: 4
  train_batch_size: 4
  
  # If `val_task` parameter is set, and thus the validation phase is performed,
  # you could change the batch size to use for the validation phase.
  # If this parameter is set to null, then the `train_batch_size` value will be used
  # as the `eval_batch_size`
  eval_batch_size: null
  
```

All parameters of the *model* section should be defined as attribute of the **model** mapping

Check the [available models](../available_implementations/available_models.md) to see which models
are implemented at the moment and their customizable parameters!
