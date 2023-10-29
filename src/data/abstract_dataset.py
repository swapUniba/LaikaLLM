from __future__ import annotations
import inspect
from collections import OrderedDict
from typing import get_type_hints

from src.parser_utils import CMDExec


class LaikaDataset:

    def __init_subclass__(cls, **kwargs):
        current_dataset_parser = CMDExec.dataset_parser.add_parser(cls.__name__)

        constructor_signature = inspect.signature(cls.__init__)
        param_annotations = get_type_hints(cls.__init__)

        all_params = OrderedDict(constructor_signature.parameters.items())

        # remove "self" parameter
        all_params.popitem(last=False)

        for param_name, param in all_params.items():

            if param.default == param.empty:
                # Parameter is mandatory
                current_dataset_parser.add_argument(f"--{param_name}",
                                                    type=param_annotations[param_name],
                                                    required=True)

                # # Handle Literal type hint
                # if param.annotation == Literal:
                #     data_parser.add_argument(f'--{param_name}', choices=param.default, required=True)
            elif param.default != param.empty:
                # Parameter is optional
                current_dataset_parser.add_argument(f'--{param_name}',
                                                    type=param_annotations[param_name],
                                                    default=param.default)

            elif param.kind == param.VAR_POSITIONAL:
                current_dataset_parser.add_argument(f'--{param_name}',
                                                    type=param_annotations[param_name],
                                                    default=param.default)

                # # Handle Literal type hint
                # if param.annotation == Literal:
                #     data_parser.add_argument(f'--{param_name}', choices=param.default)
