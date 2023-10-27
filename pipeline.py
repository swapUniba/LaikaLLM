import argparse
import os

from pygit2 import Repository

from src import ExperimentConfig
from src.data.amazon_dataset import data_main
from src.data.templates import Task
from src.evaluate.metrics import RankingMetric
from src.model.trainer import trainer_main
from src.utils import seed_everything, init_wandb, LoadFromYaml

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script to reproduce perform the experiments')
    parser.add_argument('-c', '--config', action=LoadFromYaml, const="params.yml", default=None,
                        help='')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='')
    parser.add_argument('--checkpoint', type=str, default="google/flan-t5-small",
                        help='')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='')
    parser.add_argument('--train_tasks', type=str, nargs="+", default=tuple(Task.str_alias_cls.keys()),
                        choices=list(Task.str_alias_cls.keys()),
                        help='', )
    parser.add_argument('--integer_ids', action="store_true",
                        help='')
    parser.add_argument('--items_start_from_1001', action="store_true",
                        help='')
    parser.add_argument('--inject_personalization', type=str, nargs="+", default=(),
                        choices=["train", "eval"],
                        help='',)
    parser.add_argument('--monitor_metric', type=str, default="no",
                        choices=list(RankingMetric.str_alias_cls.keys()),
                        help='',)
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='',)
    parser.add_argument('--eval_batch_size', type=int, default=2,
                        help='',)
    parser.add_argument('--add_prefix_items_users', action="store_true",
                        help='',)
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='',)
    parser.add_argument('--random_seed', type=int, default=42,
                        help='',)
    parser.add_argument('--log_wandb', action="store_true",
                        help='',)

    # will first parse args from yml file, and if same are passed via cmd,
    # those passed via cmd will prevail
    args = parser.parse_args()

    # nargs = "+" returns a list if set, we want a tuple instead in order to be type coherent
    args.inject_personalization = tuple(args.inject_personalization)
    args.train_tasks = tuple(args.train_tasks)

    # set default exp name
    if args.exp_name is None:
        # replace '/' with '_' to avoid creation of subdir (google/flan-t5-small -> google_flan-t5-small)
        args.exp_name = f"{args.checkpoint.replace('/', '_')}_{args.n_epochs}"

    if args.log_wandb:

        if 'WANDB_API_KEY' not in os.environ:
            raise ValueError('Cannot log run to wandb if environment variable "WANDB_API_KEY" is not present\n'
                             'Please set the environment variable and add the api key for wandb\n')

        if 'WANDB_ENTITY' not in os.environ:
            raise ValueError('Cannot log run to wandb if environment variable "WANDB_ENTITY" is not present\n'
                             'Please set the environment variable and add the entity for wandb logs\n')

    # translate args to dict
    # delete config yaml file since if it is used, we don't need it anymore since all params
    # are already been loaded up to this point
    del args.config
    dict_args = vars(args)

    # for this to work, each pipeline parameter should exist
    # as attribute of ExperimentConfig class
    for arg, arg_value in vars(args).items():
        assert hasattr(ExperimentConfig, arg), f"{arg} does not exist in ExperimentConfig class!"
        setattr(ExperimentConfig, arg, arg_value)

    print("Experiment configuration:")
    print(ExperimentConfig.to_string())

    # log env variables needed for reproducibility to args which will be logged to wandb
    dict_args["PYTHONHASHSEED"] = os.environ["PYTHONHASHSEED"]
    dict_args["CUBLAS_WORKSPACE_CONFIG"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]

    # log also the current active branch in which experiment is being performed
    dict_args["git_branch"] = Repository('.').head.shorthand

    with init_wandb(project="P5-Thesis", name=ExperimentConfig.exp_name, config=dict_args):

        # at start of each main phase, we re-initialize the state
        seed_everything(ExperimentConfig.random_seed)
        data_main()

        # at start of each main phase, we re-initialize the state
        seed_everything(ExperimentConfig.random_seed)
        trainer_main()

        # at start of each main phase, we re-initialize the state
        seed_everything(ExperimentConfig.random_seed)
        eval_main()
