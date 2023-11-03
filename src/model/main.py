import os

from datasets import Dataset

from src import SharedParams, MODELS_DIR
from src.data.abstract_dataset import LaikaDataset
from src.evaluate.abstract_metric import Metric
from src.model import ModelParams, LaikaModel
from src.model.trainer import RecTrainer


def model_main(shared_params: SharedParams, model_params: ModelParams, dataset_obj: LaikaDataset):
    # shared
    exp_name = shared_params.exp_name
    device = shared_params.device

    # trainer
    n_epochs = model_params.n_epochs
    train_batch_size = model_params.train_batch_size
    eval_batch_size = model_params.eval_batch_size
    monitor_metric = model_params.monitor_metric

    # model
    model_cls_name = model_params.model_cls_name
    model_kwargs = model_params.model_kwargs
    train_tasks = model_params.train_tasks
    val_task = model_params.val_task
    val_task_template_id = model_params.val_task_template_id

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
    rec_model = LaikaModel.from_string(model_cls_name, training_tasks_str=train_tasks, dataset_obj=dataset_obj,
                                       eval_task_str=val_task, eval_template_id=val_task_template_id,
                                       **model_kwargs)
    rec_model.to(device)

    output_dir = os.path.join(MODELS_DIR, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    [monitor_metric_obj] = Metric.from_string(monitor_metric)
    trainer = RecTrainer(
        rec_model=rec_model,
        n_epochs=n_epochs,
        batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        train_sampling_fn=sampling_fn,
        monitor_metric=monitor_metric_obj,
        output_dir=output_dir
    )

    best_model = trainer.train(train, validation_dataset=val)

    return best_model
