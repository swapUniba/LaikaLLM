---
hide:
  - toc
---

# General Parameters

```yaml title="General parameters"

# Name of the experiment: it will be used to create the directories
# in which the trained model and the metrics will be saved, respectively in
# "models" directory and "reports/metrics" directory at the root of the repository.
# In this case:
# * The model will be stored into "models/simple_exp"
# * The metrics results will be stored into "reports/metrics/simple_exp"
#
# Required
exp_name: simple_exp

# Device to use when training the model. Usually is "cpu" or "cuda:0".
# Use "cuda:0" to use your gpu and speed up the training phase
#
# Optional, Default: cuda:0
device: cuda:0


# The random state to use for the experiment. For each main phase (data/model/eval),
# the random state will be initialized to this particular value
#
# Optional, Default: 42
random_seed: 42


# If set to true the training and evaluation results will be logged to wandb # (1)
#
# Optional, Default: false
log_wandb: false

# If log_wandb is set to "true", you can customize the project to which the run will be logged
# with this parameter
#
# Optional, Default: null
wandb_project: null


# If set to true, only the evaluation phase will be performed. Be sure that the model exists
# at location "models/EXPERIMENT_NAME" # (2)
eval_only: false



```

1. Be sure to set at least 'WANDB_API_KEY' and 'WANDB_ENTITY' as environment variables like suggested in the [official
   documentation](https://docs.wandb.ai/guides/track/environment-variables#optional-environment-variables),
   otherwise an exception is raised!
2. *EXPERIMENT_NAME* is the value of the `exp_name` property in the .yaml file, in this case it's *simple_exp*
