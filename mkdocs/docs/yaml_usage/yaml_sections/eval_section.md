# Eval section

In the eval section, all parameters related to evaluation are specified, like the eval batch size,
the tasks and metrics to use for evaluation, etc.

All parameters of the *eval* section should be defined as attribute of the **eval** mapping

```yaml title="Eval section"
eval:
  
  # Mapping between the tasks to evaluate and the metrics to use during evaluation # (1)
  #
  # Required
  eval_tasks:
    SequentialSideInfo:
      - hit@1
      - hit@10
    RatingPredictionTask:
      - mae
      - rmse
  
  # The batch size to use during evaluation phase. If not specified, it uses
  # the `eval_batch_size` defined in the 'model' section. If `eval_batch_size` is not
  # defined in the 'model' section, then it will use the `train_batch_size` defined in 
  # the 'model' section
  #
  # Optional, Default: null
  eval_batch_size: null
  
  # If set to True, for each task, a latex table storing the results of each template,
  # will be saved along with the same results saved in CSV Format
  #
  # Optional, Default: true
  create_latex_table: true
```

1. Be sure to check [all available tasks]() and [all available metrics]()