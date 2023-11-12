import yaml

from src import GeneralParams
from src.data import DataParams
from src.evaluate import EvalParams
from src.model import ModelParams


def parse_yml_config(yml_path: str):

    with open(yml_path, "r") as f:
        yaml_args = yaml.safe_load(f)

    data_section = yaml_args.pop("data", None)
    model_section = yaml_args.pop("model", None)
    eval_section = yaml_args.pop("eval", None)

    # after popping every section, only general params remain
    general_section = yaml_args

    general_params = GeneralParams.from_parse(general_section)
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

    return general_params, data_params, model_params, eval_params
