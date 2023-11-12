import argparse
import dataclasses
import os

from pygit2 import Repository

from src.data.main import data_main
from src.evaluate.main import eval_main
from src.model.main import model_main
from src.utils import seed_everything, init_wandb

from src.yml_parse import parse_yml_config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script to reproduce perform the experiments')

    parser.add_argument('-c', '--config', default="params.yml", required=True, help='')

    # will first parse args from yml file, and if same are passed via cmd,
    # those passed via cmd will prevail
    args = parser.parse_args(["-c", "params.yml"])

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
    # also the current active branch in which experiment is being performed
    config_args = {
        "general_params": dataclasses.asdict(general_params),
        "data_params": dataclasses.asdict(data_params),
        "model_params": dataclasses.asdict(model_params),
        "eval_params": dataclasses.asdict(eval_params),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_branch": Repository('.').head.shorthand
    }

    with init_wandb(project="P5-Thesis", name=general_params.exp_name, config=config_args,
                    should_log=general_params.log_wandb):

        if not general_params.eval_only:
            # at start of each main phase, we re-initialize the state
            seed_everything(general_params.random_seed)
            data_main(general_params, data_params)

            # at start of each main phase, we re-initialize the state
            seed_everything(general_params.random_seed)
            model_main(general_params, data_params, model_params)

        # at start of each main phase, we re-initialize the state
        seed_everything(general_params.random_seed)
        eval_main(general_params, data_params, model_params, eval_params)
