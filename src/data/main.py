import os.path

from src import SharedParams, PROCESSED_DATA_DIR
from src.data import DataParams
from src.data.abstract_dataset import LaikaDataset


def data_main(shared_params: SharedParams, data_section_config: DataParams):

    dataset_cls_name = data_section_config.dataset_cls_name
    dataset_params = data_section_config.dataset_params

    ds = LaikaDataset.from_string(dataset_cls_name, **dataset_params)

    output_dir = os.path.join(PROCESSED_DATA_DIR, shared_params.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    ds.save(output_dir)

    return ds
