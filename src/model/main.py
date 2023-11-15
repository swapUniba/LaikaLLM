import os

from datasets import Dataset

from src import GeneralParams, MODELS_DIR, PROCESSED_DATA_DIR
from src.data import DataParams
from src.data.abstract_dataset import LaikaDataset
from src.evaluate.abstract_metric import LaikaMetric
from src.model import ModelParams, LaikaModel
from src.model.trainer import RecTrainer


def model_main(general_params: GeneralParams, data_params: DataParams, model_params: ModelParams):

    # general params
    exp_name = general_params.exp_name
    device = general_params.device
    log_wandb = general_params.log_wandb

    # trainer params
    n_epochs = model_params.n_epochs
    train_batch_size = model_params.train_batch_size
    eval_batch_size = model_params.eval_batch_size
    monitor_metric = model_params.monitor_metric

    # model params
    model_cls_name = model_params.model_cls_name
    model_kwargs = model_params.model_kwargs
    train_tasks = model_params.train_tasks
    train_task_selection_strat = model_params.train_task_selection_strat
    val_task = model_params.val_task
    val_task_template_id = model_params.val_task_template_id

    # load dataset created in data phase
    dataset_cls = LaikaDataset.dataset_exists(data_params.dataset_cls_name, return_bool=False)

    dataset_path = os.path.join(PROCESSED_DATA_DIR, general_params.exp_name)
    dataset_obj = dataset_cls.load(dataset_path)

    ds_dict = dataset_obj.get_hf_datasets()
    sampling_fn = dataset_obj.sample_train_sequence

    train = ds_dict["train"]
    val = ds_dict["validation"] if val_task is not None else None

    # REDUCE FOR TESTING
    # train = Dataset.from_dict(train[:100])
    # val = Dataset.from_dict(val[:100])

    # some parameters are "internal", in the sense that are used by any model implemented and are
    # not passed directly via yaml configuration (e.g., dataset_obj), others are passed via yaml configuration and
    # are forwarded to the model. The peculiarity is that via **model_kwargs, even new parameters not initially
    # considered are forwarded
    rec_model = LaikaModel.from_string(model_cls_name,
                                       training_tasks_str=train_tasks,
                                       train_task_selection_strat=train_task_selection_strat,
                                       dataset_obj=dataset_obj,
                                       eval_task_str=val_task,
                                       eval_template_id=val_task_template_id,
                                       **model_kwargs)
    rec_model.to(device)

    output_dir = os.path.join(MODELS_DIR, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    monitor_metric_obj = LaikaMetric.from_string(monitor_metric)
    trainer = RecTrainer(
        rec_model=rec_model,
        n_epochs=n_epochs,
        batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        train_sampling_fn=sampling_fn,
        monitor_metric=monitor_metric_obj,
        output_dir=output_dir,
        should_log=log_wandb
    )

    trainer.train(train, validation_dataset=val)
