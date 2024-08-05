import argparse
import dataclasses
import os

import yaml
from pygit2 import Repository, GitError

from src.data.main import data_main
from src.evaluate.main import eval_main
from src.model.main import model_main
from src.utils import seed_everything, init_wandb, IndentedDumper

from src.yml_parse import parse_yml_config


def pretty_print_configuration(config: dict):
    print(" Experiment configuration ".center(80, "*"))

    print("\n" + "-" * 80)
    print("Environment/General parameters:")
    print("-" * 80)

    env_var_keys = ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "git_branch")
    env_var_dict = {key: config[key] for key in env_var_keys}

    general_dict = config["general_params"]

    print(yaml.dump({**env_var_dict, **general_dict}, default_flow_style=False, Dumper=IndentedDumper))

    print("-" * 80)
    print("Data parameters:")
    print("-" * 80)
    print(yaml.dump(config["data_params"], default_flow_style=False, Dumper=IndentedDumper))

    print("-" * 80)
    print("Model parameters:")
    print("-" * 80)
    print(yaml.dump(config["model_params"], default_flow_style=False, Dumper=IndentedDumper))

    print("-" * 80)
    print("Eval parameters:")
    print("-" * 80)
    print(yaml.dump(config["eval_params"], default_flow_style=False, Dumper=IndentedDumper))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script to reproduce perform the experiments')

    parser.add_argument('-c', '--config', default="params.yml", required=True,
                        help='The path to the .yml file in which are specified all the experiment parameters')

    # parse yml config
    args = parser.parse_args()
    general_params, data_params, model_params, eval_params = parse_yml_config(args.config)

    if general_params.log_wandb:

        if 'WANDB_API_KEY' not in os.environ:
            raise ValueError('Cannot log run to wandb if environment variable "WANDB_API_KEY" is not present\n'
                             'Please set the environment variable and add the api key for wandb\n')

        if 'WANDB_ENTITY' not in os.environ:
            raise ValueError('Cannot log run to wandb if environment variable "WANDB_ENTITY" is not present\n'
                             'Please set the environment variable and add the entity for wandb logs\n')

    # this is the config dict that will be logged to wandb
    # apart from the params read from yml file, log env variables needed for reproducibility and
    # also the current active branch in which experiment is being performed (if the project is in a git directory)

    try:
        git_branch = Repository('.').head.shorthand
    except GitError:
        git_branch = None

    config_args = {
        "general_params": dataclasses.asdict(general_params),
        "data_params": dataclasses.asdict(data_params),
        "model_params": dataclasses.asdict(model_params),
        "eval_params": dataclasses.asdict(eval_params),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_branch": git_branch
    }

    pretty_print_configuration(config_args)

    with init_wandb(project=general_params.wandb_project, name=general_params.exp_name, config=config_args,
                    should_log=general_params.log_wandb):

        if not general_params.eval_only:

            print(" DATA ".center(80, "*"))

            # at start of each main phase, we re-initialize the state
            seed_everything(general_params.random_seed)
            data_main(general_params, data_params)

            print()  # simple newline

            print(" MODEL ".center(80, "*"))

            # at start of each main phase, we re-initialize the state
            seed_everything(general_params.random_seed)
            model_main(general_params, data_params, model_params)

            print()  # simple newline

        print(" EVAL ".center(80, "*"))

        # at start of each main phase, we re-initialize the state
        seed_everything(general_params.random_seed)
        eval_main(general_params, data_params, model_params, eval_params)
