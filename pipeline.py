import argparse
import os

from src import ExperimentConfig
from src.model.trainer import trainer_main
from src.utils import seed_everything, init_wandb

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script to reproduce perform the experiments')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='')
    parser.add_argument('--checkpoint', type=str, default="google/flan-t5-small",
                        help='')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='')
    parser.add_argument('--integer_ids', action=argparse.BooleanOptionalAction, default=False,
                        help='')
    parser.add_argument('--inject_personalization', type=str, nargs="+", default=(),
                        choices=["train", "eval"],
                        help='',)
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='',)
    parser.add_argument('--eval_batch_size', type=int, default=2,
                        help='',)
    parser.add_argument('--add_prefix_item_users', action=argparse.BooleanOptionalAction, default=False,
                        help='',)
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='',)
    parser.add_argument('--random_seed', type=int, default=42,
                        help='',)
    parser.add_argument('--log_wandb', action=argparse.BooleanOptionalAction, default=False,
                        help='',)

    args = parser.parse_args()

    # nargs = "+" returns a list if set, we want a tuple instead in order to be type coherent
    args.inject_personalization = tuple(args.inject_personalization)

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

    # for this to work, each pipeline parameter should exist
    # as attribute of ExperimentConfig class
    for arg, arg_value in vars(args).items():
        assert hasattr(ExperimentConfig, arg), f"{arg} does not exist in ExperimentConfig class!"
        setattr(ExperimentConfig, arg, arg_value)

    seed_everything(ExperimentConfig.random_seed)

    with init_wandb(project="P5-Thesis", name=ExperimentConfig.exp_name, config=args):
        trainer_main()
