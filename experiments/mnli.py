# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

# Thean Add
import self_code
# Thean End
import os
import time
import torch
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import transformers
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from contexttimer import Timer
from collections import defaultdict
from transformers import TrainingArguments
from transformers import default_data_collator
from typing import List, Dict, Tuple, Optional, Union, Any

from experiments import constants
from experiments import mnli_utils
from experiments import misc_utils
#from experiments import remote_utils
from experiments import influence_helpers
from experiments import hans
from influence_utils import nn_influence_utils
from experiments.data_utils import (
    glue_output_modes,
    glue_compute_metrics)

#MNLI_TRAINING_SCRIPT_NAME = "scripts/run_MNLI.20200913.sh"
MNLI_TRAINING_SCRIPT_NAME = "scripts/run_MNLI.20200913_xLaunch.sh"
NUM_DATAPOINTS_TO_REMOVE_CHOICES = [1, 5, 25]

CORRECT_INDICES = sorted([
    # e.g., `KNN-recall.only-correct.50.0.pth.g0301.ll.unc.edu`
    int(f.split("/")[-1].split(".")[3])  # Thean Add, this = 0 using the example above)
    for f in glob(os.path.join(
        constants.MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR,
        "*only-correct*")
    )
])
INCORRECT_INDICES = sorted([
    # e.g., `KNN-recall.only-correct.50.0.pth.g0301.ll.unc.edu`
    int(f.split("/")[-1].split(".")[3])
    for f in glob(os.path.join(
        constants.MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR,
        "*only-incorrect*")
    )
])

# Thean
#  Called by run_experiments's MNLI_retraining_experiments()
# End Thean
def run_retraining_main(
        mode: str,
        num_examples_to_test: int):

    if mode not in ["full", "KNN-1000", "KNN-10000", "random"]:
        raise ValueError(f"Unrecognized `mode` {mode}")

    for example_relative_index in range(num_examples_to_test):
        for correct_mode in ["correct", "incorrect"]:
            if correct_mode == "correct":
                example_index = CORRECT_INDICES[example_relative_index]
            if correct_mode == "incorrect":
                example_index = INCORRECT_INDICES[example_relative_index]

            if mode in ["full"]:
                # Load file from local or sync from remote
                file_name = os.path.join(
                    constants.MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR,
                    f"KNN-recall.only-{correct_mode}.50.{example_index}"
                    f".pth.g0301.ll.unc.edu")

                influences_dict = torch.load(file_name)
                if example_index != influences_dict["test_index"]:
                    raise ValueError

                if (correct_mode == "correct" and
                        influences_dict["correct"] is not True or
                        correct_mode == "incorrect" and
                        influences_dict["correct"] is True):
                    raise ValueError

                helpful_indices, harmful_indices = (
                    misc_utils.get_helpful_harmful_indices_from_influences_dict(
                        influences_dict["influences"]))

                indices_dict = {
                    "helpful": helpful_indices,
                    "harmful": harmful_indices}

            if mode in ["KNN-1000", "KNN-10000"]:
                if mode == "KNN-1000":
                    kNN_k = 1000
                if mode == "KNN-10000":
                    kNN_k = 10000

                file_name = os.path.join(
                    constants.MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR2,
                    f"visualization"
                    f".only-{correct_mode}"
                    f".5.mnli-mnli-None-mnli"
                    f".{kNN_k}.True.pth.g0306.ll.unc.edu")

                influences_dict = torch.load(file_name)[example_relative_index]
                if example_index != influences_dict["index"]:
                    raise ValueError

                helpful_indices, harmful_indices = (
                    misc_utils.get_helpful_harmful_indices_from_influences_dict(
                        influences_dict["influences"]))

                indices_dict = {
                    "helpful": helpful_indices,
                    "harmful": harmful_indices}

            if mode == "random":
                # Get indices corresponding to each label
                label_to_indices = mnli_utils.get_label_to_indices_map()
                np.random.shuffle(label_to_indices["neutral"])
                np.random.shuffle(label_to_indices["entailment"])
                np.random.shuffle(label_to_indices["contradiction"])
                indices_dict = {
                    "neutral": label_to_indices["neutral"],
                    "entailment": label_to_indices["entailment"],
                    "contradiction": label_to_indices["contradiction"],
                }

            for tag, indices in indices_dict.items():
                for num_data_points_to_remove in NUM_DATAPOINTS_TO_REMOVE_CHOICES:
                    if len(indices) < num_data_points_to_remove:
                        raise ValueError(f"`indices` have only {len(indices)} elements "
                                         f"whereas {num_data_points_to_remove} is needed")

                    run_one_retraining(
                        indices=indices[:num_data_points_to_remove],
                        dir_name=(
                            f"./retraining-remove-"
                            f"{example_index}-"
                            f"{correct_mode}-"
                            f"{mode}-"
                            f"{tag}-"
                            f"{num_data_points_to_remove}"))


def run_one_retraining(
        indices: List[int],
        dir_name: str,
) -> None:
    mnli_utils.create_one_set_of_data_for_retraining(
        dir_name=dir_name,
        indices_to_remove=indices)
    output_dir = os.path.join(dir_name, "output_dir")
    subprocess.check_call([
        "bash",
        MNLI_TRAINING_SCRIPT_NAME,
        dir_name, output_dir
    ])
    # client = remote_utils.ScpClient()
    # client.scp_file_to_remote(
    #     local_file_name=dir_name,
    #     remote_file_name=os.path.join(
    #         constants.REMOTE_DEFAULT_REMOTE_BASE_DIR,
    #         f"{dir_name}.{client.host_name}"),
    #     # This is a folder
    #     recursive=True)

# Thean
#  Called by run_experiments's KNN_recall_experiments
#
#  Note: we are studying the influence of a training datapoint z on the test datapoint ztest
#        "the influence function refers to the change in the model’s loss on the test data-point ztest if we
#         up-weight the loss of training data-point z by epsilon"
# End Thean
def run_full_influence_functions(
        mode: str,
        num_examples_to_test: int,
        s_test_num_samples: int = 1000
) -> Dict[int, Dict[str, Any]]:

    if mode not in ["only-correct", "only-incorrect"]:
        raise ValueError(f"Unrecognized mode {mode}")

    tokenizer, model = misc_utils.create_tokenizer_and_model(
        constants.MNLI_MODEL_PATH)

    (mnli_train_dataset,
     mnli_eval_dataset) = misc_utils.create_datasets(
        task_name="mnli",
        tokenizer=tokenizer)

    batch_train_data_loader = misc_utils.get_dataloader(
        mnli_train_dataset,
        batch_size=128,
        random=True)

    instance_train_data_loader = misc_utils.get_dataloader(
        mnli_train_dataset,
        batch_size=1,
        random=False)

    eval_instance_data_loader = misc_utils.get_dataloader(
        dataset=mnli_eval_dataset,
        batch_size=1,
        random=False)

    output_mode = glue_output_modes["mnli"]

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
        model=model,
        args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100),
        data_collator=default_data_collator,
        train_dataset=mnli_train_dataset,
        eval_dataset=mnli_eval_dataset,
        compute_metrics=build_compute_metrics_fn("mnli"),
    )

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    model.cuda()
    num_examples_tested = 0
    outputs_collections = {}
    
    # Thean Add
    folder = "experiments_outputs/"
    fn = f"KNN-recall.{mode}.{num_examples_to_test}.{s_test_num_samples}.collections.pth"

    path = self_code.checkfile(folder+fn)
    fn = self_code.after_str(path, folder)

    if "_" in fn:
        ext = self_code.between_str(fn, "_", ".pth")
        print(ext)
        ext = int(ext)
        next_starting_test_index = (ext + 1) * num_examples_to_test * 3
        # Ex: num_examples_to_test = 3; and I ran the code once before
        # then, my ext == 0 (see how checkfile works), next_starting_test_index = 3
        # Adding in the * 2 to hopefully wish the model at most predict incorrectly
        # twice the num_examples_to_test at one time
    else:
        next_starting_test_index = 0
     
    torch.save('temp', path)
    # End Thean
    
    for test_index, test_inputs in enumerate(eval_instance_data_loader):
        # Thean ADD
        if test_index < next_starting_test_index:
            continue
        # Ex: num_examples_to_test = 3; and I ran the code once before
        # then next_starting_test_index = 3, and the previously used 
        # test_index are 0, 1, 2. We want to start at least from next_starting_test_index
        # assuming ALL of the previous prediction matches the "mode"
        # This is not a foolproof method so you need to do some extra runs
        # to account for overlaps on test_index
        # End Thean
        
        # Thean: 
        #  A.1: For each evaluation data-point ztest,
        # End Thean
        if num_examples_tested >= num_examples_to_test:
            break

        # Skip when we only want cases of correction prediction but the
        # prediction is incorrect, or vice versa
        # Thean:
        #  A.1 We select 100 data-points from
        #  the MNLI evaluation dataset (50 data-points when
        #  the model predictions are correct, 50 when they
        #  are incorrect) and aggregate the results.
        # End Thean
        prediction_is_correct = misc_utils.is_prediction_correct(
            trainer=trainer,
            model=model,
            inputs=test_inputs)

        if mode == "only-correct" and prediction_is_correct is False:
            continue

        if mode == "only-incorrect" and prediction_is_correct is True:
            continue
        
        # Thean
        #  A.1 we first compute the ground-truth influential
        #  data-points via running influence functions
        #  on the MultiNLI training dataset without
        #  kNN (i.e., { top-m influential }).
        # End Thean

        with Timer() as timer:
            influences, _, s_test = nn_influence_utils.compute_influences(
                n_gpu=1,
                device=torch.device("cuda"),
                batch_train_data_loader=batch_train_data_loader,       #Thean: batch_size of 128 (fastif would use 1), randomSeq
                instance_train_data_loader=instance_train_data_loader, #Thean: batch_size of 1, fixSeq
                model=model,
                test_inputs=test_inputs,
                params_filter=params_filter,
                weight_decay=constants.WEIGHT_DECAY,
                weight_decay_ignores=weight_decay_ignores,
                s_test_damp=5e-3,
                s_test_scale=1e4,
                s_test_num_samples=s_test_num_samples,                 #Thean: default arg = 1000, J
                train_indices_to_include=None,
                s_test_iterations=1,                                   #Thean: T
                precomputed_s_test=None)

            outputs = {
                "test_index": test_index,
                "influences": influences,  #Thean: the index of the influences correspond to training data indices"
                "s_test": s_test,
                "time": timer.elapsed,
                "correct": prediction_is_correct,
            }
            num_examples_tested += 1
            outputs_collections[test_index] = outputs
            '''
            remote_utils.save_and_mirror_scp_to_remote(
                object_to_save=outputs,
                file_name=f"KNN-recall.{mode}.{num_examples_to_test}.{test_index}.pth")
            '''
            #Thean adds code to save
            '''
            torch.save(
                outputs,
                f"KNN-recall.{mode}.{num_examples_to_test}.{test_index}.pth")
            
            print(f"Status: #{test_index} | {num_examples_tested} / {num_examples_to_test}")
            '''
            #Thean adds code to save 
    torch.save(
        outputs_collections, path)
    
    return outputs_collections

# Thean
#  Called by run_experiment.imitator_experiments
# End Thean
def imitator_main(mode: str, 
                  num_examples_to_test: int,
                  #Thean Add
                  similarity: str = "feature",
                  metric: str = "L2",
                  direction: str = "similar"
                  #Thean End
                 ) -> List[Dict[str, Any]]:
    if mode not in ["only-correct", "only-incorrect"]:
        raise ValueError(f"Unrecognized mode {mode}")
        
    # Thean Add    
    if similarity not in ['feature', 'pred_feature']:
        raise ValueError("Choose similarity from `feature` or `pred_feature`")
    
    if metric not in ["L2", "inner_product", "cosine_similarity"]:
        raise ValueError("Choose metric from `L2`, `inner_product`, or `cosine_similarity`")
    
    if direction not in ["similar", "dissimilar", "mixed"]:
        raise ValueError("Choose direction from `similar`, `dissimilar`, or `mixed`")
    # Thean End

    task_tokenizer, task_model = misc_utils.create_tokenizer_and_model(
        constants.MNLI_MODEL_PATH)

    imitator_tokenizer, imitator_model = misc_utils.create_tokenizer_and_model(
        constants.MNLI_IMITATOR_MODEL_PATH)

    (mnli_train_dataset,
     mnli_eval_dataset) = misc_utils.create_datasets(
        task_name="mnli",
        tokenizer=task_tokenizer)

    task_model.cuda()
    imitator_model.cuda()
    if task_model.training is True or imitator_model.training is True:
        raise ValueError("One of the model is in training mode")
    print(task_model.device, imitator_model.device)

    # Most of these arguments are placeholders
    # and are not really used at all, so ignore
    # the exact values of these.
    trainer = transformers.Trainer(
        model=task_model,
        args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100),
    )
    
    # Thean Add: random: bool = False for misc.get_dataloader()
    eval_instance_data_loader = misc_utils.get_dataloader(
        mnli_eval_dataset,
        batch_size=1,
        data_collator=default_data_collator)
    
    train_inputs_collections = torch.load(
        constants.MNLI_TRAIN_INPUT_COLLECTIONS_PATH)
    inputs_by_label: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(train_inputs_collections)):
        label = mnli_train_dataset.label_list[
            train_inputs_collections[i]["labels"]]
        inputs_by_label[label].append(i)
    # Thean's version - deprecated
    # inputs_by_label = mnli_utils.get_label_to_indices_map()
    # End Thean - deprecated

    outputs_collections = []
    #Thean Add
    num_examples_tested = 0
    # End Thean
    for i, test_inputs in enumerate(eval_instance_data_loader):
        
        # Thean modify
        '''
        if mode == "only-correct" and i not in CORRECT_INDICES[:num_examples_to_test]:
            continue
        if mode == "only-incorrect" and i not in INCORRECT_INDICES[:num_examples_to_test]:
            continue
        '''
        if num_examples_tested >= num_examples_to_test:
            break
            
        prediction_is_correct = misc_utils.is_prediction_correct(
            trainer=trainer,
            model=task_model,
            inputs=test_inputs)

        if mode == "only-correct" and prediction_is_correct is False:
            continue

        if mode == "only-incorrect" and prediction_is_correct is True:
            continue
        # End Thean Modify

        start_time = time.time()
        for using_ground_truth in [True, False]:
            outputs = run_one_imitator_experiment(
                task_model=task_model,
                imitator_model=imitator_model,
                test_inputs=test_inputs,
                trainer=trainer,
                train_dataset=mnli_train_dataset,
                train_inputs_collections=train_inputs_collections,
                inputs_by_label=inputs_by_label,
                finetune_using_ground_truth_label=using_ground_truth,
                #Thean Add
                similarity = similarity,
                metric = metric,
                direction = direction
                #Thean End
                )
            outputs["index"] = i
            outputs_collections.append(outputs)

        end_time = time.time()
        print(f"#{len(outputs_collections)}/{len(outputs_collections)}: "
              f"Elapsed {(end_time - start_time) / 60:.2f}")
        
        #Thean Add
        num_examples_tested+=1
        # Thean End
    
    fn = "./experiments_outputs/"+f"imitator_experiments.{mode}.{num_examples_to_test}.pth"
    fn = self_code.checkfile(fn)
    torch.save(
        outputs_collections, fn)

    return outputs_collections

# Thean Add
#  See B.1
#  Note that we use k = 10000.
#  All it does is creating data_indices and data_tags, and pass everything to compute_new_imitator_losses()
#  Called by imitator_main()
#  Calls:
#     _make_imitator_inputs()
#    influence_helpers' .load_faiss_index(), .select_s_test_config(), .compute_influences_simplified()
# End Thean
def run_one_imitator_experiment(
    task_model: torch.nn.Module,     
    imitator_model: torch.nn.Module, 
    test_inputs,                     
    trainer: transformers.Trainer,
    train_dataset: torch.utils.data.Dataset,
    train_inputs_collections: List,      #Thean Add: List[Dict[str, torch.Tensor]]      
    inputs_by_label: Dict[str, List[int]],  
    sample_size: int = 10,                  
    num_nearest_neighbors: int = 10000,    
    # Thean Add: B.1 Footnote 18: We experimented with both settings, and found using the original/true labels
    #   performed better
    # Thean End
    finetune_using_ground_truth_label: bool = False,
    # Thean Start
    similarity: str = "feature",
    metric: str = "L2",
    direction: str = "similar"
    # Thean End
) -> Dict[str, Any]:
    '''
    Thean Add
    1) Create imitator_test_inputs which is test_inputs with task_model's PREDICTED LABEL
    2) Calculate influences of `num_nearest_neighbors` training data onto `test_inputs`
    3) Create data_indices and corresponding tags. Influences is used in part of the data_indices creation.
    4) compute_new_imitator_losses().
    
    
    task_model:               model trained on the original training data
    imitator_model:           model trained on the predictions from the task_model
    test_inputs:              this is used to calculate influences, which impact which most harmful/helpful training
                              data being used in fine-tuning the imitator model across various learning rates
    train_dataset:            pass into compute_influences_simplified() to compute influences only.
    train_inputs_collections: training data: List[Dict[str, torch.Tensor]]
    inputs_by_label:          used to randomly select `sample_size` counts of train_index by label/tags
    sample_size:              number of training data per label/tags that should be used by the `imitator_model` to
                              fine-tune on. This should be 10 according to the paper `mean performance averaged across
                              10 fine tuning data-points`
    num_nearest_neighbors:    number of training data to be filtered and used to calculate their indv influences 
                              on the test_inputs
    '''
    # Thean Add    
    if similarity not in ['feature', 'pred_feature']:
        raise ValueError("Choose similarity from `feature` or `pred_feature`")
    
    if metric not in ["L2", "inner_product", "cosine_similarity"]:
        raise ValueError("Choose metric from `L2`, `inner_product`, or `cosine_similarity`")
    
    if direction not in ["similar", "dissimilar", "mixed"]:
        raise ValueError("Choose direction from `similar`, `dissimilar`, or `mixed`")
    # Thean End
    
    
    #Thean: this returns test_inputs with taskModel predicted Ylabel
    imitator_test_inputs = _make_imitator_inputs(
        trainer=trainer, task_model=task_model, inputs=test_inputs)
    # if labels[0] != logits.argmax(axis=1)[0]:
    #     break
    faiss_index = influence_helpers.load_faiss_index(
        trained_on_task_name="mnli",
        train_task_name="mnli",
        #Thean Add
        similarity = similarity
        #Thean End
    )

    s_test_damp, s_test_scale, s_test_num_samples = influence_helpers.select_s_test_config(
        trained_on_task_name="mnli",
        train_task_name="mnli",
        eval_task_name="mnli")
    
    # Thean Add
    # Need to update this function to support pred_feature similarity
    # influence = Dict{train_index, influence_value}
    # This value is used to pick the most harm/helpful indices.
    # Thean End
    influences = influence_helpers.compute_influences_simplified(
        k=num_nearest_neighbors,
        faiss_index=faiss_index,
        model=task_model,
        inputs=test_inputs,
        train_dataset=train_dataset,
        use_parallel=False,
        s_test_damp=s_test_damp,               # Thean: select_s_test_config CONSTANT
        s_test_scale=s_test_scale,             # Thean: select_s_test_config CONSTANT
        s_test_num_samples=s_test_num_samples, # Thean: J. select_s_test_config CONSTANT
        similarity = similarity,
        metric = metric,
        direction = direction)
    
    # Thean Add
    # Randomly pick "sample_size" counts of "neutral", "entailment" and "contradiction" data_indices
    # Pick the top "sample_size" positive and negative influential data_indices
    # Convert all of them into lists, concatenate them and put them into a tuple ()
    # The data_indices are later used in compute_new_imitator_losses() to create 
    # imitator_train_inputs, either by extracting relevant training data using data_indices
    # OR generate data using _make_imitator_inputs().
    # Thean End
    data_indices = (
        np.random.choice(inputs_by_label["neutral"],  #Thean Add: inputs_by_label: Dict[str, List[train_index]]
                         size=sample_size,
                         replace=False).tolist() +  # noqa
        np.random.choice(inputs_by_label["entailment"],
                         size=sample_size,
                         replace=False).tolist() +  # noqa
        np.random.choice(inputs_by_label["contradiction"],
                         size=sample_size,
                         replace=False).tolist() +  # noqa
        misc_utils.sort_dict_keys_by_vals(influences)[:sample_size] +  # noqa
        misc_utils.sort_dict_keys_by_vals(influences)[-sample_size:]
    )
    
    # Thean
    # B.1 Details:  " choose five types of data for fine-tuning"
    # Create corresponding data tags to match the above data_indices
    # The tag is later used to index the imitator_losses by tag
    # End Thean
    data_tags = (
        ["random-neutral" for _ in range(sample_size)] +  # noqa
        ["random-entailment" for _ in range(sample_size)] +  # noqa
        ["random-contradiction" for _ in range(sample_size)] +  # noqa
        ["most-negative-influential" for _ in range(sample_size)] +  # noqa
        ["most-positive-influential" for _ in range(sample_size)]
    )
    
    # Thean
    #  B.1 Details: fine-tune on 1 data-point with 50 learning rates in log-space from 10−5 to 10−2.5
    # End Thean
    learning_rates = np.logspace(-5, -2.5, 50)
    # Thean Add: 
    # losses  = Dict[data_tags, [Train5[losses_Lrate1, losses_lrate2,..], Train9[losses_Lrate1, losses_lrate2,..]]
    # Thean End
    losses = compute_new_imitator_losses(
        trainer=trainer,
        tags=data_tags,
        indices=data_indices,                               #Thean Add:"Training data" indices for the imitator_model
        task_model=task_model,
        imitator_model=imitator_model,                      
        learning_rates=learning_rates,                      #Thean: np.logspace(-5, -2.5, 50)
        imitator_test_inputs=imitator_test_inputs,          #Thean: for predicting, calculating imitator model loss
        train_inputs_collections=train_inputs_collections,  #Thean Add:"Traning data" for the imitator_model
        finetune_using_ground_truth_label=finetune_using_ground_truth_label)

    return {
        "losses": losses,
        "influences": influences,                      # Thean Add: influence = Dict{train_index, influence_value}
        "test_inputs": test_inputs,                    # Thean Add: This is used to compute influences
        "learning_rates": learning_rates,              # Thean Add: np.logspace(-5, -2.5, 50)
        "imitator_test_inputs": imitator_test_inputs,  # Thean Add: for predicting, calculating imitator model loss
        "finetune_using_ground_truth_label": finetune_using_ground_truth_label,
    }

# Thean Add
# This is called by run_one_imitator_experiment().
# It uses 
#  _make_imitator_inputs()
#  hans.pseudo_gradient_step() to fine tune the imitator model
#  misc_utils.predict() to get imitator prediction loss
# Thean End
def compute_new_imitator_losses(
        indices: List[int],
        tags: List[str],
        task_model: torch.nn.Module,
        imitator_model: torch.nn.Module,
        trainer: transformers.Trainer,
        learning_rates: Union[np.ndarray, List[float]],  
        imitator_test_inputs: Dict[str, torch.Tensor], 
        train_inputs_collections: List[Dict[str, torch.Tensor]],
        finetune_using_ground_truth_label: bool = False, 
) -> Dict[str, List[List[float]]]:
    '''
    Thean Add
    For each `indices, tags` pair:
      1) Generate imitator_train_inputs. This includes training data Xs with a Label Y. Label Y could be real, or it 
             could be based on task_model prediction.
      2) For each learning rate:
          a) Fine tune the `imitator_model`
          b) Calculate it's prediction loss w.r.t. `imitator_test_inputs`: the `task_model` prediction on
              `test_inputs`.
    
    indices:                           Training data indices from various labels/tags/classes
    tags:                              Corresponding to `indices`. Includes random-neutral, random-entailment,
                                       random-contradiction, most-negative-influential, most-positive-influential
    learning_rates:                    np.logspace(-5, -2.5, 50)
    imitator_test_inputs:              for predicting, calculating imitator model loss. The Y LABEL is based on
                                       task_model prediction, not the ground-truth
    finetune_using_ground_truth_label: mentioned at B.1 Footnote 18 We experimented with both settings, and found
                                       using the original/true labels performed better
    '''
    params_filter = [
        n for n, p in imitator_model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in imitator_model.named_parameters()
        if not p.requires_grad]

    losses = defaultdict(list)  # Thean Add: turns a list into a dictionary
    # Thean Add: Pairing up indices and its tags created by run_one_imitator_experiment()
    # Example tags are random-neutral, random-entailment
    # End Thean
    for index, tag in zip(tqdm(indices), tags):           #Thean: Usually total of 50 loops per article
        if finetune_using_ground_truth_label is True:
            imitator_train_inputs = train_inputs_collections[index]
        else:
            #Thean Add: This returns task model training inputs and its predicted Y label
            imitator_train_inputs = _make_imitator_inputs(
                trainer=trainer,
                task_model=task_model,
                inputs=train_inputs_collections[index])

        _losses = []
        gradients_z = None
        
        # Thean
        #  B.1 Details: 
        #   For a given test data-point and fine-tuning type, we fine-tune on 1 data-point with 50 learning rates
        #   in log-space from 10−5 to 10−2.5 , and repeat for 10 different fine-tuning data-points
        # End Thean
        for lr in learning_rates:
            # Re-use `gradients_z`
            new_imitator_model, gradients_z = hans.pseudo_gradient_step(
                model=imitator_model,
                inputs=imitator_train_inputs,
                learning_rate=lr,
                #params_filter=params_filter,  #Thean remove this because the func defn doesnt need it
                #weight_decay_ignores=weight_decay_ignores, #Thean remove this because as above
                precomputed_gradients_z=gradients_z)
            
            _, _, imitator_loss = misc_utils.predict(
                trainer=trainer,
                model=new_imitator_model,
                inputs=imitator_test_inputs)
            
            _losses.append(imitator_loss)

        losses[tag].append(_losses)

    return losses  # Thean Add: losses of the imitator model w.r.t. task_model Prediction Y Label

# Thean Add
# This is called by 
#  - run_one_imitator_experiment() and 
#  - compute_new_imitator_losses()
# Thean End
def _make_imitator_inputs(
        trainer: transformers.Trainer,
        task_model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    '''
    Thean Add:
    Returns inputs with task_model's PREDICTED Y LABELS using the inputs
    '''
    logits, _, _ = misc_utils.predict(
        trainer=trainer, model=task_model, inputs=inputs)
    imitator_inputs = deepcopy(inputs)
    imitator_inputs["labels"] = torch.tensor(logits.argmax(axis=1))
    return imitator_inputs

# Thean Add
# This plots Figure 5
# Thean End
def plot_Xs_and_Ys_dict(
        Xs: List[float],
        Ys_dict: Dict[str, List[List[float]]]
) -> None:
    # plt.rcParams["figure.figsize"] = (10, 10)
    color_map = {
        "random-neutral": "grey",
        "random-entailment": "salmon",
        "random-contradiction": "skyblue",
        "most-positive-influential": "darkred",
        "most-negative-influential": "steelblue"}

    legends = []
    for tag in Ys_dict.keys():
        if tag not in color_map.keys():
            raise ValueError(tag)

        legends.append(tag)
        color = color_map[tag]
        data = np.array(Ys_dict[tag])
        is_random_data_point = "random" in tag

        if data.shape[0] != 1:
            data_mean = data.mean(axis=0)
            data_max = data.max(axis=0)
            data_min = data.min(axis=0)
            plt.plot(Xs, data_mean,
                     color=color,
                     linestyle=("--" if is_random_data_point else None))

            plt.fill_between(Xs, data_max, data_min,
                             alpha=0.1,
                             color=color)
        else:
            plt.plot(Xs, data[0, ...], color=color)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("learning rate", fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.legend(legends, fontsize=15, loc=(1.04,0.5))
    plt.title("Loss of the Imitator Model", fontsize=30)
