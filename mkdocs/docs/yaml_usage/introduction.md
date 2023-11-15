# Yaml usage

Using LaikaLLM via `.yaml` is really simple:

- You define the steps of your experiment following the **data-model-evaluation** logic
- You invoke LaikaLLM with the following command:

```bash
python LaikaLLM.py -c params.yml # (1)
```

1. Instead of params.yml, specify the path to your .yaml file

!!! info

    The necessary requirement for using LaikaLLM with the *user-defined* .yaml configuration is to set the root of the
    repository as the **Working Directory**

## Yaml interface

The .yaml file parameters can be grouped into *four* different macro-section:

1. The first section contains **[general parameters](yaml_sections/general_parameters.md)** needed by all the other sections, like 
   the *experiment name*, the *random state*, the *device* to use, etc.
2. The second section contains all the parameters needed by the **[data phase](yaml_sections/data_section.md)** of the experiment, 
   such as the *dataset* to use and its parameters
3. The third section contains all the parameters needed by the **[model phase](yaml_sections/model_section.md)** of the experiment, 
   such as the *model* to use, the *number of epochs* to train the model, the *train batch size*, etc.
4. The fourth section contains all the parameters needed for the **[eval phase](yaml_sections/eval_section.md)** of the experiment, 
   such as the *eval batch size*, the *metrics* to use, etc.

You can check each subsection to see all its customizable parameters!
