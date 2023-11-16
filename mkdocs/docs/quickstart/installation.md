# Installation

*LaikaLLM* requires **Python 3.10** or later, and all packages needed are listed in 
[`requirements.txt`](https://github.com/Silleellie/LaikaLLM/blob/main/requirements.txt)

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

**NOTE**: It is **highly** suggested to set the following environment variables to obtain *100%* reproducible results of
your experiments:

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
```

You can check useful info about the above environment variables [here](https://docs.python.org/3.3/using/cmdline.html#envvar-PYTHONHASHSEED) and [here](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)

!!! info

    *LaikaLLM* can be easily run by defining a `.yaml` file, which encapsulates the full logic of the experiment,
    or by accessing the Python API (a more *flexible* and *powerful* approach).
    Check the [`.yaml`](simple_yaml_example.md)
    sample example or the [`python`](simple_python_example.md) one to get up and running with LaikaLLM!


---
**Tip**: It is suggested to install LaikaLLM requirements in a virtual environment

!!! quote ""
    *Virtual environments are special isolated environments where all the packages and versions you install only 
    apply to that specific environment. It’s like a private island! — but for code.*

Read this [Medium article][medium] for understanding all the advantages and the [official python guide] [venv]
on how to set up one

[medium]: https://towardsdatascience.com/why-you-should-use-a-virtual-environment-for-every-python-project-c17dab3b0fd0
[venv]: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/