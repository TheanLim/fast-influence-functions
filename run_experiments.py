# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import sys
from typing import Optional, Dict

from experiments import mnli
from experiments import hans
from experiments import s_test_speedup
#from experiments import remote_utils
#from experiments import visualization


USE_PARALLEL = False
# Thean:
#  A.1: We select 100 data-points from
#  the MNLI evaluation dataset (50 data-points when
#  the model predictions are correct, 50 when they
#  are incorrect) and aggregate the results.
# End Thean
NUM_KNN_RECALL_EXPERIMENTS = 3#50  

# Thean
#  A.3 Details: "(3 for data-points where the prediction is correct, and 3 where it is incorrect)"
# End Thean
NUM_RETRAINING_EXPERIMENTS = 1#3

# Thean
# A.2  .. repeat the experiments for 8 different MNLI evaluation data points
# (4 when the prediction is correct, 4 when the prediction is incorrect)
# End Thean
NUM_STEST_EXPERIMENTS = 1#10   # It should be 4 based on A.2

NUM_VISUALIZATION_EXPERIMENTS = 1#100

# Thean
#  Figure 9 uses 20 test data-points
#  Figure 5 only uses 4 test data points
#
# NUM_IMITATOR_EXPERIMENTS = 10 means that you will use 10 correct prediction test data and 10 incorrrect
#   summing up to 20 test data points per author
# End Thean
NUM_IMITATOR_EXPERIMENTS = 10


def KNN_recall_experiments(
        mode: str,
        num_experiments: Optional[int] = None
) -> None:
    """Experiments to Check The Influence Recall of KNN"""
    print("RUNNING `KNN_recall_experiments`")

    if num_experiments is None:
        num_experiments = NUM_KNN_RECALL_EXPERIMENTS

    # (a) when the prediction is correct, and (b) incorrect
    mnli.run_full_influence_functions(
        mode=mode,
        num_examples_to_test=num_experiments)


def s_test_speed_quality_tradeoff_experiments(
        mode: str,
        num_experiments: Optional[int] = None
) -> None:
    """Experiments to Check The Speed/Quality Trade-off of `s_test` estimation"""
    print("RUNNING `s_test_speed_quality_tradeoff_experiments`")

    if num_experiments is None:
        num_experiments = NUM_STEST_EXPERIMENTS

    # (a) when the prediction is correct, and (b) incorrect
    s_test_speedup.main(
        mode=mode,
        num_examples_to_test=num_experiments,
        num_repetitions = 1)#4)  # Thean Modified. Let's try and see if this will cut down running time <8 hours


def MNLI_retraining_experiments(
        mode: str,
        num_experiments: Optional[int] = None
) -> None:
    print("RUNNING `MNLI_retraining_experiments`")

    if num_experiments is None:
        num_experiments = NUM_RETRAINING_EXPERIMENTS

    mnli.run_retraining_main(
        mode=mode,
        num_examples_to_test=num_experiments)


def visualization_experiments(
        num_experiments: Optional[int] = None
) -> None:
    """Experiments for Visualizing Effects"""
    print("RUNNING `visualization_experiments`")

    if num_experiments is None:
        num_experiments = NUM_VISUALIZATION_EXPERIMENTS

    for heuristic in hans.DEFAULT_EVAL_HEURISTICS:
        visualization.main(
            train_task_name="hans",
            eval_task_name="hans",
            num_eval_to_collect=num_experiments,
            use_parallel=USE_PARALLEL,
            hans_heuristic=heuristic,
            trained_on_task_name="hans")

    visualization.main(
        train_task_name="hans",
        eval_task_name="mnli-2",
        num_eval_to_collect=num_experiments,
        use_parallel=USE_PARALLEL,
        hans_heuristic=None,
        trained_on_task_name="hans")


def hans_augmentation_experiments(
        num_replicas: Optional[int] = None
) -> None:
    print("RUNNING `hans_augmentation_experiments`")
    # We will use the all the `train_heuristic` here, as we did in
    # `eval_heuristics`. So looping over the `DEFAULT_EVAL_HEURISTICS`
    # Thean add: each mnli-2 and hans take 8 hours.
    # The model used is trained on mnli-2, fine-tuned on `train_task_name`, 
    # and evaluated on HANS. See Figure 6(a) and 6(b) - v2 paper. OR figure 7 of the v1 paper
    # End Thean
    for train_task_name in ["mnli-2"]:#, "mnli-2"]: 
        #for train_heuristic in hans.DEFAULT_EVAL_HEURISTICS: # Thean: ["lexical_overlap", "subsequence", "constituent"]
         for train_heuristic in ["lexical_overlap"]: #Thean
            for version in ["new-only-z"]:#, "new-only-ztest", "new-z-and-ztest"]:
                # Thean: each inner loop takes 3++ hours if not using parallel
                hans.main(
                    train_task_name=train_task_name,
                    train_heuristic=train_heuristic,
                    num_replicas=num_replicas,
                    use_parallel=USE_PARALLEL,
                    version=version,
                    #Thean Add
                    similarity = "pred_feature"
                    #Thean End
                )


def imitator_experiments(
        num_experiments: Optional[int] = None,
    # Thean Start
        similarity: str = "feature",
        metric: str = "L2"
        # Thean End
) -> None:
    print("RUNNING `imitator_experiments`")
    
    # Thean Add
    if similarity not in ['feature', 'pred_feature']:
        raise ValueError("Choose similarity from `feature` or `pred_feature`")
    
    if metric not in ["L2", "inner_product", "cosine_similarity"]:
        raise ValueError("Choose metric from `L2`, `inner_product`, or `cosine_similarity`")
    # Thean End

    if num_experiments is None:
        num_experiments = NUM_IMITATOR_EXPERIMENTS
    
    mnli.imitator_main(
        mode="only-correct",
        num_examples_to_test=num_experiments,
        #Thean Add
        similarity = similarity,
        metric = metric,
        direction = "mixed"
        # End Thean
    )
    

    mnli.imitator_main(
        mode="only-incorrect",
        num_examples_to_test=num_experiments,
        #Thean Add
        similarity = similarity,
        metric = metric,
        direction = "mixed"
        # End Thean
    )

if __name__ == "__main__":
    # Make sure the environment is properly setup
    #remote_utils.setup_and_verify_environment()

    experiment_name = sys.argv[1]
    if experiment_name == "knn-recall-correct":
        KNN_recall_experiments(
            mode="only-correct")
    if experiment_name == "knn-recall-incorrect":
        KNN_recall_experiments(
            mode="only-incorrect")

    if experiment_name == "s-test-correct":
        s_test_speed_quality_tradeoff_experiments(
            mode="only-correct")
    if experiment_name == "s-test-incorrect":
        s_test_speed_quality_tradeoff_experiments(
            mode="only-incorrect")

    if experiment_name == "retraining-full":
        MNLI_retraining_experiments(
            mode="full")

    if experiment_name == "retraining-random":
        MNLI_retraining_experiments(
            mode="random")

    if experiment_name == "retraining-KNN-1000":
        MNLI_retraining_experiments(
            mode="KNN-1000")

    if experiment_name == "retraining-KNN-10000":
        MNLI_retraining_experiments(
            mode="KNN-10000")

    if experiment_name == "hans-augmentation":
        hans_augmentation_experiments()

    if experiment_name == "imitator":
        imitator_experiments(
            similarity = "pred_feature",
            metric = "inner_product"
        )

    #raise ValueError(f"Unknown Experiment Name: {experiment_name}")
