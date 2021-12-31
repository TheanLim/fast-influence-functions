# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
import numpy as np
import transformers
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from transformers import InputFeatures
from transformers import default_data_collator
from typing import Union, Dict, Any, List, Tuple, Optional

from influence_utils import faiss_utils
from influence_utils import nn_influence_utils
from experiments import constants
from experiments import misc_utils
# from experiments import remote_utils
from experiments import influence_helpers
from experiments.hans_utils import HansHelper
from transformers import TrainingArguments
from experiments.data_utils import (
    glue_output_modes,
    glue_compute_metrics,
    CustomGlueDataset)

'''
Table 3:
Experiments are repeated 3 times, fine-tuning is applied repeatedly for 10 iterations. 
For each iteration, we select 10 validation data-points as the “anchor” data-points, 
update parameters for one gradient step on 10 fine-tuning data-points with learning rate 10−4
'''


DEFAULT_KNN_K = 1000
# Thean: Repeat Exp 3 times
DEFAULT_NUM_REPLICAS = 3 
# Thean: 10 validation data-points as the "anchor" data-points
EVAL_HEURISTICS_SAMPLE_BATCH_SIZE = 10
# Thean Add"
# C.3 Details: "we compare the performance between using helpful data-points, harmful data-points, and random
#               data-points."
# Thean End
EXPERIMENT_TYPES = ["most-helpful", "most-harmful", "random"]
# Thean Add"
# When the models are evaluated on HANS, we use each of the three slices of
# the HANS dataset as the evaluation dataset.
# Thean End
DEFAULT_EVAL_HEURISTICS = ["lexical_overlap", "subsequence", "constituent"]
VERSION_2_NUM_DATAPOINTS_CHOICES = [EVAL_HEURISTICS_SAMPLE_BATCH_SIZE]
# Thean Add
# C.3 Details 
# ... and update model parameters for one gradient step on 10 fine-tuning
#     data-point with learning rate 10−4
# Thean End
VERSION_2_LEARNING_RATE_CHOICES = [1e-4]

# Thean Add:  
# trained on A, fine-tuned on B, and then evaluated on C
# train_task_name is the B
def main(
        train_task_name: str,  # Thean ["mnli-2", "hans"]
        train_heuristic: str,  # Thean one from ["lexical_overlap", "subsequence", "constituent"]
        eval_heuristics: Optional[List[str]] = None,
        num_replicas: Optional[int] = None,
        use_parallel: bool = True,
        version: Optional[str] = None,
        #Thean Add
        similarity:str = "feature"
        #Thean End
) -> Dict[str, List[Dict[str, Any]]]:

    if train_task_name not in ["mnli-2", "hans"]:
        raise ValueError

    if eval_heuristics is None:
        eval_heuristics = DEFAULT_EVAL_HEURISTICS  # Thean: ["lexical_overlap", "subsequence", "constituent"]

    if num_replicas is None:
        num_replicas = DEFAULT_NUM_REPLICAS # Thean: 3

    if version not in ["new-only-z", "new-only-ztest", "new-z-and-ztest"]:
        raise ValueError

    task_tokenizer, task_model = misc_utils.create_tokenizer_and_model(
        constants.MNLI2_MODEL_PATH)

    (mnli_train_dataset,
     mnli_eval_dataset) = misc_utils.create_datasets(
        task_name="mnli-2",
        tokenizer=task_tokenizer)

    (hans_train_dataset,
     hans_eval_dataset) = misc_utils.create_datasets(
        task_name="hans",
        tokenizer=task_tokenizer)

    if train_task_name == "mnli-2":
        train_dataset = mnli_train_dataset

    if train_task_name == "hans":
        train_dataset = hans_train_dataset
    
    # Thean Add
    #    s_test_damp = 5e-3
    #    s_test_scale = 1e6
    #    s_test_num_samples = 1000
    # End Thean
    (s_test_damp,
     s_test_scale,
     s_test_num_samples) = influence_helpers.select_s_test_config(
        trained_on_task_name="mnli-2",
        train_task_name=train_task_name,
        eval_task_name="hans",
    )
    # Thean Add:  ^^^
    # trained on A, fine-tuned on B, and then evaluated on C
    # trained_on_task_name = A
    # train_task_name = B
    # eval_task_name = C
    

    hans_helper = HansHelper(
        hans_train_dataset=hans_train_dataset,
        hans_eval_dataset=hans_eval_dataset)

    # We will be running model trained on MNLI-2
    # but calculate influences on HANS dataset  # Thean Add HANS or mnli-2 dataset
    faiss_index = influence_helpers.load_faiss_index(
        trained_on_task_name="mnli-2",
        train_task_name=train_task_name,
        #Thean Add
        similarity =similarity  
        #Thean End
    )

    output_mode = glue_output_modes["mnli-2"]  # output_mode = "classification"
    
    # Thean: This is a function generator; but it is not used at all?!
    def build_compute_metrics_fn(task_name: str):
        def compute_metrics_fn(p):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Most of these arguments are placeholders
    # and are not really used at all, so ignore
    # the exact values of these.
    trainer = transformers.Trainer(
        model=task_model,  # Thean: i.e. mnli2
        args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100),
    )
    
    # Thean Add: 
    # See: Using List as default_factory section of https://www.geeksforgeeks.org/defaultdict-in-python/
    output_collections: Dict[str, List] = defaultdict(list)

    if version == "old":
        raise ValueError("Deprecated")

    else:
        # Thean Add
        # C.3 Details
        # We repeat the Step 2-4 for 10 iterations. 
        # Thean: This is the number of "fine-tuning model"
        # Thean End
        NUM_STEPS = 10
        num_total_experiments = (
            len(EXPERIMENT_TYPES) *  # Thean: ["most-helpful", "most-harmful", "random"]
            num_replicas * # Thean i.e. 3
            len(VERSION_2_NUM_DATAPOINTS_CHOICES) * # Thean: i.e. 1; VERSION_2_NUM_DATAPOINTS_CHOICES = [10] 
            len(VERSION_2_LEARNING_RATE_CHOICES) *  # Thean i.e. 1
            NUM_STEPS
        ) # Thean default num_total_experiments = 90
        
        #with tqdm(total=num_total_experiments, disable=True) as pbar: 
        with tqdm(total=num_total_experiments, disable=True) as pbar:
            for experiment_type in EXPERIMENT_TYPES:  # Thean: ["most-helpful", "most-harmful", "random"]
                for replica_index in range(num_replicas):  # Thean i.e. 3
                    
                    # Thean add: we get different and randomly chosen 10 data on each iteration
                    # Sample anchor data-points every step
                    (hans_eval_heuristic_inputs,
                     hans_eval_heuristic_raw_inputs) = hans_helper.sample_batch_of_heuristic(
                        mode="eval",
                        heuristic=train_heuristic,  # Thean i.e. one of ["lexical_overlap", "subsequence", "constituent"]
                        size=EVAL_HEURISTICS_SAMPLE_BATCH_SIZE,  # Thean i.e. 10
                        return_raw_data=True)

                    misc_utils.move_inputs_to_device(
                        inputs=hans_eval_heuristic_inputs,
                        device=task_model.device)   # Thean: i.e. task_model = mnli2

                    for version_2_num_datapoints in VERSION_2_NUM_DATAPOINTS_CHOICES: #Thean: i.e. 10
                        for version_2_learning_rate in VERSION_2_LEARNING_RATE_CHOICES: # Thean i.e. 1e-4

                            # The SAME model will be used for multiple
                            # steps so `deepcopy` it here.
                            # Thean: _model is not carried over for each iteration.
                            _model = deepcopy(task_model)
                            for step in range(NUM_STEPS):  # Thean i.e. 10
                                outputs_one_experiment, _model = one_experiment(
                                    use_parallel=use_parallel,
                                    train_heuristic=train_heuristic, #Thean oneOf ["lexical_overlap", "subsequence", "constituent"]
                                    eval_heuristics=eval_heuristics, # Thean i.e. ["lexical_overlap", "subsequence", "constituent"]
                                    experiment_type=experiment_type, # Thean: one of ["most-helpful", "most-harmful", "random"]
                                    hans_helper=hans_helper,
                                    train_dataset=train_dataset,     # Thean: hans_train_dataset or mnli_train_dataset
                                    task_model=_model,               # Thean: mnli-2 model
                                    faiss_index=faiss_index,
                                    s_test_damp=s_test_damp,         # Thean: i.e. 5e-3
                                    s_test_scale=s_test_scale,       # Thean: i.e. 1e6
                                    s_test_num_samples=s_test_num_samples, #Thean i.e. 1000
                                    trainer=trainer,
                                    version=version,              #Thean oneOf ["new-only-z", "new-only-ztest", "new-z-and-ztest"]
                                    version_2_num_datapoints=version_2_num_datapoints, #Thean: i.e. 10
                                    version_2_learning_rate=version_2_learning_rate,   #Thean: i.e. 1e-4
                                    hans_eval_heuristic_inputs=hans_eval_heuristic_inputs,          #Thean: i.e. size of 10
                                    hans_eval_heuristic_raw_inputs=hans_eval_heuristic_raw_inputs,  #Thean: i.e. size of 10
                                    #Thean Add
                                    similarity = similarity
                                    #Thean End
                                )

                                output_collections[
                                    f"{experiment_type}-"
                                    f"{replica_index}-"
                                    f"{version_2_num_datapoints}-"
                                    f"{version_2_learning_rate}-"
                                ].append(outputs_one_experiment)

                                pbar.update(1)
                                pbar.set_description(f"{experiment_type} #{replica_index}")

        torch.save(
            output_collections,
            f"hans-augmentation-{version}."
            f"{train_task_name}."
            f"{train_heuristic}."
            f"{num_replicas}."
            f"{use_parallel}_{similarity}.pth")  #Thean Add PREDFEAT

    return output_collections


def one_experiment(
    use_parallel: bool,
    train_heuristic: str,       # Thean One of["lexical_overlap", "subsequence", "constituent"]
    eval_heuristics: List[str], # Thean i.e. ["lexical_overlap", "subsequence", "constituent"]
    experiment_type: str,       # Thean: one of ["most-helpful", "most-harmful", "random"]
    hans_helper: HansHelper,
    train_dataset: CustomGlueDataset,    # Thean: hans_train_dataset or mnli_train_dataset
    task_model: torch.nn.Module,         # Thean: mnli-2 model
    faiss_index: faiss_utils.FAISSIndex,
    s_test_damp: float,                  # Thean: i.e. 5e-3
    s_test_scale: float,                 # Thean: i.e. 1e6
    s_test_num_samples: int,             # Thean: i.e. 1000
    trainer: transformers.Trainer,
    version: str,                        #Thean oneOf ["new-only-z", "new-only-ztest", "new-z-and-ztest"]
    version_2_num_datapoints: Optional[int],  #Thean: i.e. 10
    version_2_learning_rate: Optional[float], #Thean: i.e. 1e-4
    hans_eval_heuristic_inputs: Dict[str, Any],           #Thean: i.e. size of 10
    hans_eval_heuristic_raw_inputs: List[InputFeatures],  #Thean: i.e. size of 10
    #Thean Add
    similarity:str = "feature"
    #Thean End
) -> Tuple[Dict[str, Any], Optional[torch.nn.Module]]:
    if task_model.device.type != "cuda":
        raise ValueError("The model is supposed to be on CUDA")

    if version_2_num_datapoints is None:
        raise ValueError
    if version_2_learning_rate is None:
        raise ValueError

    if experiment_type in ["most-harmful", "most-helpful"]:  # Thean: one of ["most-helpful", "most-harmful"]
        
        if similarity == "pred_feature":
            if experiment_type == "most-harmful":
                direction = "dissimilar"
            elif experiment_type == "most-helpful":
                direction = "similar"
        elif similarity == "feature":
            direction = "similar"

        influences = influence_helpers.compute_influences_simplified(
            k=DEFAULT_KNN_K,
            faiss_index=faiss_index,
            model=task_model,
            inputs=hans_eval_heuristic_inputs,
            train_dataset=train_dataset,
            use_parallel=use_parallel,
            s_test_damp=s_test_damp,
            s_test_scale=s_test_scale,
            s_test_num_samples=s_test_num_samples,
            device_ids=[1, 2, 3],
            precomputed_s_test=None,
            faiss_index_use_mean_features_as_query=True,
            #Thean Add
            similarity= similarity,
            metric= "inner_product" if similarity == "pred_feature" else "L2",
            direction= direction
            # Thean End
        )
        helpful_indices, harmful_indices = misc_utils.get_helpful_harmful_indices_from_influences_dict(
            influences)#Thean, n=version_2_num_datapoints)
        if experiment_type == "most-helpful":
            datapoint_indices = helpful_indices[:version_2_num_datapoints]

        if experiment_type == "most-harmful":
            datapoint_indices = harmful_indices[:version_2_num_datapoints]

    if experiment_type == "random":
        # s_test = None
        influences = None
        hans_eval_heuristic_inputs = None
        # Essentially shuffle the indices
        datapoint_indices = np.random.choice(
            len(train_dataset),
            size=len(train_dataset),
            replace=False)
    
    print(experiment_type, ": ", datapoint_indices)
    loss_collections = {}
    accuracy_collections = {}

    # num_datapoints = 1
    # learning_rate = 1e-4
    num_datapoints = version_2_num_datapoints  #Thean: i.e. 10
    learning_rate = version_2_learning_rate    #Thean: i.e. 1e-4

    if version == "new-only-z":
        datapoints = [
            train_dataset[index]                               #Thean: train_dataset = hans_train_dataset or mnli_train_dataset
            for index in datapoint_indices[:num_datapoints]]   #Thean: i.e. num_datapoints = 10

    if version == "new-only-ztest":
        datapoints = hans_eval_heuristic_raw_inputs            #Thean: i.e. size of 10

    if version == "new-z-and-ztest":
        datapoints = [
            train_dataset[index]
            for index in datapoint_indices[:num_datapoints]   #Thean: i.e. num_datapoints = 10
        ] + hans_eval_heuristic_raw_inputs

    batch = default_data_collator(datapoints)
    new_model, _ = pseudo_gradient_step(
        model=task_model,            # Thean: mnli-2 model
        inputs=batch,
        learning_rate=learning_rate) # Thean: i.e. 1e-4

    for heuristic in eval_heuristics:  # Thean i.e. ["lexical_overlap", "subsequence", "constituent"]
        new_model_loss, new_model_accuracy = evaluate_heuristic(
            hans_helper=hans_helper,
            heuristic=heuristic,
            trainer=trainer,
            model=new_model)            # Thean new_model is the mnli-2 model with pseudo-gradient step applied

        loss_collections[heuristic] = new_model_loss
        accuracy_collections[heuristic] = new_model_accuracy
        # print(f"Finished {num_datapoints}-{learning_rate}")

    output_collections = {
        # "s_test": s_test,
        "influences": influences,
        "loss": loss_collections,
        "accuracy": accuracy_collections,
        "datapoint_indices": datapoint_indices,
        "learning_rate": learning_rate,
        "num_datapoints": num_datapoints,
        "hans_eval_heuristic_inputs": hans_eval_heuristic_inputs,
    }

    # Warning: Check again whether using this `new_model` is a good idea
    return output_collections, new_model


def pseudo_gradient_step(
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        learning_rate: float,
        precomputed_gradients_z: Optional[List[torch.FloatTensor]] = None
) -> Tuple[torch.nn.Module, List[torch.FloatTensor]]:

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    params_to_freeze = [
        "bert.embeddings.",
        "bert.encoder.layer.0.",
        "bert.encoder.layer.1.",
        "bert.encoder.layer.2.",
        "bert.encoder.layer.3.",
        "bert.encoder.layer.4.",
        "bert.encoder.layer.5.",
        "bert.encoder.layer.6.",
        "bert.encoder.layer.7.",
        "bert.encoder.layer.8.",
        "bert.encoder.layer.9.",
    ]

    if precomputed_gradients_z is not None:
        gradients_z = precomputed_gradients_z
    else:
        gradients_z = nn_influence_utils.compute_gradients(
            n_gpu=1,
            device=torch.device("cuda"),
            model=model,
            inputs=inputs,
            params_filter=params_filter,
            weight_decay=constants.WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores)

    new_model = deepcopy(model)
    params_to_update = [
        p for name, p in new_model.named_parameters()
        if not any(pfreeze in name for pfreeze in params_to_freeze)]

    # They should refer to the same parameters
    if len(params_to_update) != len(gradients_z):
        raise ValueError

    with torch.no_grad():
        [p.sub_(learning_rate * grad_z) for p, grad_z in
         zip(params_to_update, gradients_z)]

    return new_model, gradients_z


def evaluate_heuristic(
        hans_helper: HansHelper,
        heuristic: str,
        trainer: transformers.Trainer,
        model: torch.nn.Module,
) -> Tuple[float, float]:

    _, batch_dataloader = hans_helper.get_dataset_and_dataloader_of_heuristic(
        mode="eval",
        heuristic=heuristic,
        batch_size=1000,
        random=False)

    loss = 0.
    num_corrects = 0.
    num_examples = 0
    for index, inputs in enumerate(batch_dataloader):
        batch_size = inputs["labels"].shape[0]
        batch_preds, batch_label_ids, batch_mean_loss = misc_utils.predict(
            trainer=trainer,
            model=model,
            inputs=inputs)

        num_examples += batch_size
        loss += batch_mean_loss * batch_size
        num_corrects += (batch_preds.argmax(axis=-1) == batch_label_ids).sum()

    return loss / num_examples, num_corrects / num_examples

def create_FAISS_index(
    train_task_name: str,
    trained_on_task_name: str
) -> faiss_utils.FAISSIndex:
    
    if train_task_name not in ["mnli-2", "hans"]:
        raise ValueError

    if trained_on_task_name not in ["mnli-2", "hans"]:
        raise ValueError
    
    if trained_on_task_name == "mnli-2":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.MNLI2_MODEL_PATH)

    if trained_on_task_name == "hans":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.HANS_MODEL_PATH)

    train_dataset, _ = misc_utils.create_datasets(
        task_name=train_task_name,
        tokenizer=tokenizer)
    
    faiss_index = faiss_utils.FAISSIndex(768, "Flat")
    
    model.cuda()
    device = model.device
    train_batch_data_loader = misc_utils.get_dataloader(
        dataset=train_dataset,
        batch_size=128,
        random=False)

    for inputs in tqdm(train_batch_data_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        features = misc_utils.compute_BERT_CLS_feature(model, **inputs)
        features = features.cpu().detach().numpy()
        faiss_index.add(features)

    return faiss_index

'''
# Create a function that returns (pred - actual)*features
# Thean Add
import numpy as np
import self_code
# Thean End

def create_FAISS_index_pred_feat(
    train_task_name: str,
    trained_on_task_name: str
) -> faiss_utils.FAISSIndex:
    if train_task_name not in ["mnli", "mnli-2", "hans"]:
        raise ValueError

    if trained_on_task_name not in ["mnli", "mnli-2", "hans"]:
        raise ValueError
    
    # Thean Add
    if trained_on_task_name == "mnli":
        # MNLI uses 3 labels for Y; The hans and mnli-2 uses two labels
        if train_task_name != "mnli":
            raise ValueError
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.MNLI_MODEL_PATH)
    # Thean End
        
    if trained_on_task_name == "mnli-2":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.MNLI2_MODEL_PATH)

    if trained_on_task_name == "hans":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.HANS_MODEL_PATH)

    train_dataset, _ = misc_utils.create_datasets(
        task_name=train_task_name,
        tokenizer=tokenizer)
    
    model.cuda()
    device = model.device
    
    train_batch_data_loader = misc_utils.get_dataloader(
        dataset=train_dataset,
        batch_size=128, 
        random=False)
    
    # Thean ADD
    trainer = transformers.Trainer(
        model=model,
        args=TrainingArguments(
        output_dir="./tmp-output",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=5e-5,
        logging_steps=100),
    )
    
    # 768+1 because we add an additional "1" for the Bias term
    # multiply num_labels because of outer product
    faiss_index = faiss_utils.FAISSIndex(trainer.model.num_labels*(768+1), 
                                         "Flat", 
                                         inner_product_metrics = True)
    # Thean End

    for inputs in tqdm(train_batch_data_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        
        # Thean START
        pred_features_vec = self_code.pred_feature_sim(model = model, 
                                                       inputs = inputs, 
                                                       trainer = trainer)
        # Thean END
        faiss_index.add(pred_features_vec) 
        
    return faiss_index
'''