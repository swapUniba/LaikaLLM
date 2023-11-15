# Data section

In the data section, the only attribute to specify is the *dataset* to use along with its parameters, like this:

```yaml title="Data section"
data:
  DATASET_TO_USE:
    PARAM1: VAL1
    PARAM2:
      - VAL2
      - VAL3
    ...

```

All parameters of the *data* section should be defined as attribute of the **data** mapping

Check the [available datasets](../available_implementations/available_datasets.md) to see which datasets
are implemented at the moment and their customizable parameters!
