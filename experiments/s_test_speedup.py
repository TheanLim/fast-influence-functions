# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

# Thean Add
import self_code
# for naming path and filename properly
from experiment_config import *
# Thean End

import sys
import torch
import transformers
import numpy as np
from contexttimer import Timer
from typing import List, Dict, Any
from transformers import GlueDataset
from transformers import TrainingArguments
from transformers import default_data_collator

from influence_utils import parallel
from influence_utils import faiss_utils
from influence_utils import nn_influence_utils
from influence_utils.nn_influence_utils import compute_s_test
from experiments import constants
from experiments import misc_utils
#from experiments import remote_utils
from experiments.data_utils import (
    glue_output_modes,
    glue_compute_metrics)


def one_experiment(
        model: torch.nn.Module,
        train_dataset: GlueDataset,
        test_inputs: Dict[str, torch.Tensor],
        batch_size: int,
        random: bool,
        n_gpu: int,
        device: torch.device,
        damp: float,
        scale: float,
        num_samples: int,
) -> List[torch.Tensor]:
    '''
    Thean Add
    
    batch_size:   B in the paper
    num_samples:  J in the paper
    random:       Whether to randomize `batch_train_data_loader` or not. Usually yes.
    '''

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    # Make sure each dataloader is re-initialized
    # Thean add: train_data RANDOM seq used to estimate s_test (HVP)
    batch_train_data_loader = misc_utils.get_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,  #Thean: B in the paper
        random=random)
    
    # Thean: s_test is dependent on ztest (test data)
    s_test = compute_s_test(
        n_gpu=n_gpu,
        device=device,
        model=model,
        test_inputs=test_inputs,
        train_data_loaders=[batch_train_data_loader],  #  Thean: train_data RANDOM seq used to estimate s_test (HVP); with batch_size
        params_filter=params_filter,
        weight_decay=constants.WEIGHT_DECAY,
        weight_decay_ignores=weight_decay_ignores,
        damp=damp,
        scale=scale,
        num_samples=num_samples)                       # Thean: J  in the paper

    return [X.cpu() for X in s_test]

# Thean
#  Called by run_experiments' s_test_speed_quality_tradeoff_experiments()
#  Check section 5.2
# End Thean
def main(
    mode: str,
    num_examples_to_test: int = 5,  # Thean Add: I thought it's 8 evaluation data points?
    num_repetitions: int = 4, # T = {1,2,3,4}
) -> List[Dict[str, Any]]:

    if mode not in ["only-correct", "only-incorrect"]:
        raise ValueError(f"Unrecognized mode {mode}")

    task_tokenizer, task_model = misc_utils.create_tokenizer_and_model(
        constants.MNLI_MODEL_PATH)
    train_dataset, eval_dataset = misc_utils.create_datasets(
        task_name="mnli",
        tokenizer=task_tokenizer)
    eval_instance_data_loader = misc_utils.get_dataloader(
        dataset=eval_dataset,
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
        model=task_model,
        args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100),
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn("mnli"),
    )

    task_model.cuda()
    num_examples_tested = 0
    output_collections = []
    
    '''
    # Thean Add
    folder = "experiments_outputs/"
    fn = f"stest.{mode}.{num_examples_to_test}.{num_repetitions}.collections.pth"
    
    path = self_code.checkfile(folder+fn)
    fn = self_code.after_str(path, folder)

    if "_" in fn:
        ext = self_code.between_str(fn, "_", ".pth")
        print(ext)
        ext = int(ext)
        next_starting_test_index = (ext + 1) * num_examples_to_test * 3
        # Ex: num_examples_to_test = 3; and I ran the code once before
        # then, my ext == 0 (see how checkfile works), next_starting_test_index = 3
        # Adding in the * 3 to hopefully wish the model at most predict incorrectly
        # twice the num_examples_to_test at one time
    else:
        next_starting_test_index = 0
     
    torch.save('temp', path)
    # End Thean
    '''
    
    # Thean ADD
    prev_simulation_count, prev_test_index = get_vars(mode)
    folder = "experiments_outputs/"
        
    if (prev_simulation_count == MAX_SIM_COUNT)or (prev_simulation_count ==0):
        # Set prev_simulation_count to one
        cur_simulation_count = 1
    else:
        cur_simulation_count = prev_simulation_count + 1
    
    fn = f"stest.{mode}.{num_examples_to_test}.{num_repetitions}.{cur_simulation_count}.collections.pth"
    path = self_code.checkfile(folder+fn)
    
    set_vars(mode, cur_simulation_count = cur_simulation_count)
    torch.save('temp', path)

    # Thean End
    
    for test_index, test_inputs in enumerate(eval_instance_data_loader):
                
        if num_examples_tested >= num_examples_to_test:
            break
        
        # Thean Start        
        # cur_simulation_count==1 Means its a whole new example (aka the first simulation)
        if cur_simulation_count ==1:
            # Re-get the test_index in case it has changed on other parallel runs
            _, prev_test_index = get_vars(mode)  # reload_config() is called within get_vars()
            next_starting_test_index = prev_test_index + 1 
        else:
            next_starting_test_index = prev_test_index
                 
        if test_index < next_starting_test_index:
            continue
        #OLD Comments 
        #Ex: num_examples_to_test = 3; and I ran the code once before
        # then next_starting_test_index = 3, and the previously used 
        # test_index are 0, 1, 2. We want to start at least from next_starting_test_index
        # assuming ALL of the previous prediction matches the "mode"
        # This is not a foolproof method so you need to do some extra runs
        # to account for overlaps on test_index
        # End Thean

        # Skip when we only want cases of correction prediction but the
        # prediction is incorrect, or vice versa
        prediction_is_correct = misc_utils.is_prediction_correct(
            trainer=trainer,
            model=task_model,
            inputs=test_inputs)

        if mode == "only-correct" and prediction_is_correct is False:
            continue

        if mode == "only-incorrect" and prediction_is_correct is True:
            continue
        
        # Thean Add
        # Set vars if we are doing computation on new test data...
        if cur_simulation_count == 1:
            set_vars(mode, cur_test_index= test_index)
        # End Thean
        
        for k, v in test_inputs.items():
            if isinstance(v, torch.Tensor):
                test_inputs[k] = v.to(torch.device("cuda"))

        # with batch-size 128, 1500 iterations is enough
        # Thean Add
        #  See Figure 4
        # Thean End
        for num_samples in range(700, 1300 + 1, 100):  # 7 choices  #Thean: J
            for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:  # 8 choices # Thean" B = {2^0, 2^1, ..., 2^7}
                for repetition in range(num_repetitions):    # Thean: T independent repetitions
                    print(f"Running #{test_index} "
                          f"N={num_samples} "
                          f"B={batch_size} "
                          f"R={repetition} takes ...", end=" ")
                    with Timer() as timer:
                        s_test = one_experiment(
                            model=task_model,
                            train_dataset=train_dataset,
                            test_inputs=test_inputs,
                            batch_size=batch_size,                # Thean: B
                            random=True,
                            n_gpu=1,
                            device=torch.device("cuda"),
                            damp=constants.DEFAULT_INFLUENCE_HPARAMS["mnli"]["mnli"]["damp"],
                            scale=constants.DEFAULT_INFLUENCE_HPARAMS["mnli"]["mnli"]["scale"],
                            num_samples=num_samples)              # Thean: J
                        time_elapsed = timer.elapsed
                        print(f"{time_elapsed:.2f} seconds")

                    outputs = {
                        "test_index": test_index,
                        "num_samples": num_samples,
                        "batch_size": batch_size,
                        "repetition": repetition,
                        "s_test": s_test,
                        "time_elapsed": time_elapsed,
                        "correct": prediction_is_correct,
                    }
                    output_collections.append(outputs)
                    '''
                    remote_utils.save_and_mirror_scp_to_remote(
                        object_to_save=outputs,
                        file_name=f"stest.{mode}.{num_examples_to_test}."
                                  f"{test_index}.{num_samples}."
                                  f"{batch_size}.{repetition}.pth")
                    '''
                    '''
                    #Thean adds code to save output
                    torch.save(
                        outputs,
                        f"stest.{mode}.{num_examples_to_test}."
                        f"{test_index}.{num_samples}."
                        f"{batch_size}.{repetition}.pth")
                    # Thean End
                    '''

        num_examples_tested += 1
    
    torch.save(
        output_collections,
        path)
        #f"stest.{mode}.{num_examples_to_test}."
        #f"{num_repetitions}.collections.pth")
    
    return output_collections