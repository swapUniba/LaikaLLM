import yaml

from src import SharedParams
from src.data import DataParams
from src.data.main import data_main
from src.evaluate import EvalParams
from src.evaluate.main import eval_main
from src.model import ModelParams
from src.model.main import model_main


def parse_yml_config(yml_path: str):

    with open(yml_path, "r") as f:
        yaml_args = yaml.safe_load(f)

    data_section = yaml_args.pop("data", None)
    model_section = yaml_args.pop("model", None)
    eval_section = yaml_args.pop("eval", None)

    # after popping every section, only general params remain
    general_section = yaml_args

    shared_params = SharedParams.from_parse(general_section)
    data_params = DataParams.from_parse(data_section)
    model_params = ModelParams.from_parse(model_section)
    eval_params = EvalParams.from_parse(eval_section)

    # copy the eval batch size from the model params
    # in case eval batch size in the eval section is not specified, just so that
    # the user doesn't need to specify it twice
    # (this takes into account the fact the if eval_batch_size in model params is None, it is set
    # to train_batch_size)
    if eval_params.eval_batch_size is None:
        eval_params.eval_batch_size = model_params.eval_batch_size

    # If the user doesn't specify eval tasks, then we will evaluate all train
    # tasks
    if eval_params.eval_tasks is None:
        eval_params.eval_tasks = model_params.train_tasks

    return shared_params, data_params, model_params, eval_params
