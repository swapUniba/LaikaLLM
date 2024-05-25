# Other Experiments

This directory contains all the `.yml` config files and results of additional experiment runs by varying the `train_task_selection_strategy`
hyperparameter and the training tasks provided by *LaikaLLM*.
All the following results make use of the new set of prompts defined that can be found [here](https://silleellie.github.io/LaikaLLM/yaml_usage/available_implementations/available_tasks/).

Each result directory contains a table storing metrics results for each task in both `.csv` and `.tex` format, generated with *LaikaLLM*.

All runs have been tracked with **WandB**. The full workspace is available by clicking the following image:

<p align="center">
  <a href="https://wandb.ai/silleellie/LaikaLLM" > 
    <img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-logo-yellow-dots-black-wb.svg" alt="Visualize runs in WandB workspace" width="20%"/>
  </a>
</p>


## Experiment 'random'

These are the results of T5-S, FlanT5-S, FlanT5-B, GPT2, with (***+W***) and without personalization, when trained and evaluated on the *Sequential*, *Direct* and *Rating Prediction* tasks
with `train task selection strategy: random`. The evaluation is carried out on all prompts, already *seen* by the model during the fine-tuning phase,
and in the following table there are reported the best results for each metric achieved by any prompt of the specific task (*best-seen*).
<p align="center">
    <img src="exp_random_results.png" alt="Experiment 'random' results"/>
</p>

### T5 runs

- **T5-S (r)**: [[.yml config](exp_random/exp_random_t5_s.yml)][[Results directory](exp_random/metrics_results/exp_random_t5_s)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/bxw0sogy)]
- **T5-S + W (r)**: [[.yml config](exp_random/exp_random_t5_s+w.yml)][[Results directory](exp_random/metrics_results/exp_random_t5_s+w)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/he9ypxdb)]

### Flan T5 Runs

- **FlanT5-S (r)**: [[.yml config](exp_random/exp_random_flan_t5_s.yml)][[Results directory](exp_random/metrics_results/exp_random_flan_t5_s)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/3kpr4nas)]
- **FlanT5-S + W (r)**: [[.yml config](exp_random/exp_random_flan_t5_s+w.yml)][[Results directory](exp_random/metrics_results/exp_random_flan_t5_s+w)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/7n03oosq)]
- **FlanT5-B (r)**: [[.yml config](exp_random/exp_random_flan_t5_b.yml)][[Results directory](exp_random/metrics_results/exp_random_flan_t5_b)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/x8fmesms)]
- **FlanT5-B + W (r)**: [[.yml config](exp_random/exp_random_flan_t5_b+w.yml)][[Results directory](exp_random/metrics_results/exp_random_flan_t5_b+w)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/efba0es8)]

### GPT2 Runs

- **GPT2 (r)**: [[.yml config](exp_random/exp_random_gpt2.yml)][[Results directory](exp_random/metrics_results/exp_random_gpt2)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/0kc6nh03)]
- **GPT2 + W (r)**: [[.yml config](exp_random/exp_random_gpt2+w.yml)][[Results directory](exp_random/metrics_results/exp_random_gpt2+w)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/sve0j31i)]


## Experiment 'only sequential'

These are the results of T5-S, FlanT5-S, FlanT5-B, GPT2, with (***+W***) and without personalization, when trained and evaluated **only** on the *Sequential* task.
The evaluation is carried out on all prompts, already *seen* by the model during the fine-tuning phase,
and in the following table there are reported the best results for each metric achieved by any prompt of the specific task (*best-seen*).

<p align="center">
    <img src="exp_seq_results.png" alt="Experiment 'only sequential' results" width="70%"/>
</p>

### T5 runs

- **T5-S**: [[.yml config](exp_seq/exp_seq_t5_s.yml)][[Results directory](exp_seq/metrics_results/exp_seq_t5_s)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/jv3dqr74)]
- **T5-S + W**: [[.yml config](exp_seq/exp_seq_t5_s+w.yml)][[Results directory](exp_seq/metrics_results/exp_seq_t5_s+w)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/4y3idqj5)]


### Flan T5 Runs

- **FlanT5-S**: [[.yml config](exp_seq/exp_seq_flan_t5_s.yml)][[Results directory](exp_seq/metrics_results/exp_seq_flan_t5_s)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/2t521nr4)]
- **FlanT5-S + W**: [[.yml config](exp_seq/exp_seq_flan_t5_s+w.yml)][[Results directory](exp_seq/metrics_results/exp_seq_flan_t5_s+w)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/cuz0hguh)]
- **FlanT5-B**: [[.yml config](exp_seq/exp_seq_flan_t5_b.yml)][[Results directory](exp_seq/metrics_results/exp_seq_flan_t5_b)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/f5l9lzm5)]
- **FlanT5-B + W**: [[.yml config](exp_seq/exp_seq_flan_t5_b+w.yml)][[Results directory](exp_seq/metrics_results/exp_seq_flan_t5_b+w)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/our9jkc3)]


### GPT2 Runs

- **GPT2**: [[.yml config](exp_seq/exp_seq_gpt2.yml)][[Results directory](exp_seq/metrics_results/exp_seq_gpt2)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/jzmgz2p9)]
- **GPT2 + W**: [[.yml config](exp_seq/exp_seq_gpt2+w.yml)][[Results directory](exp_seq/metrics_results/exp_seq_gpt2+w)][[Visualize in WandB](https://wandb.ai/silleellie/LaikaLLM/runs/dnw1ope5)]
