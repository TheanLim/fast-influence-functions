# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
from influence_utils import faiss_utils
from typing import List, Dict, Tuple, Optional, Union, Any

from experiments import constants
from experiments import misc_utils
from influence_utils import parallel
from influence_utils import nn_influence_utils

# Thean Add
import faiss
import self_code
# Thean End


def load_faiss_index(
        trained_on_task_name: str,
        train_task_name: str,
        # Thean Add
        similarity:str = "feature"#,
        #metric:str = "L2"
        # End Thean
) -> faiss_utils.FAISSIndex:
    
    '''
    similarity: feature similarity    or    prediction and feature similarity
    '''

    if trained_on_task_name not in ["mnli", "mnli-2", "hans"]:
        raise ValueError

    if train_task_name not in ["mnli", "mnli-2", "hans"]:
        raise ValueError
        
    # Thean Add    
    if similarity not in ['feature', 'pred_feature']:
        raise ValueError("Choose similarity from `feature` or `pred_feature`")
        
    #if metric not in ["L2", "inner_product", "cosine_similarity"]:
    #    raise ValueError("Choose metric from `L2`, `inner_product`, or `cosine_similarity`")
    
    # Thean Add
    if similarity =="feature":
    # End Thean
    
    # Thean Add
    # There's no need to specify any metric to load the index
    # they should be specified if one created faiss indices with > 1 
    # kind of metrics per similarity.
    # Specifying metric ensures we load the correct faiss indices
    # Thean End
        faiss_index = faiss_utils.FAISSIndex(768, "Flat") #, metric=metric)
        
        if trained_on_task_name == "mnli" and train_task_name == "mnli":
            faiss_index.load(constants.MNLI_FAISS_INDEX_PATH)
        elif trained_on_task_name == "mnli-2" and train_task_name == "mnli-2":
            faiss_index.load(constants.MNLI2_FAISS_INDEX_PATH)
        elif trained_on_task_name == "hans" and train_task_name == "hans":
            faiss_index.load(constants.HANS_FAISS_INDEX_PATH)
        elif trained_on_task_name == "mnli-2" and train_task_name == "hans":
            faiss_index.load(constants.MNLI2_HANS_FAISS_INDEX_PATH)
        elif trained_on_task_name == "hans" and train_task_name == "mnli-2":
            faiss_index.load(constants.HANS_MNLI2_FAISS_INDEX_PATH)
        else:
            faiss_index = None
    # Thean Add
    elif similarity =="pred_feature":
        
        # 768 is the features length
        # For pred_feature: 768+1 because we add an additional "1" for the Bias term
        #                   multiply num_labels because of outer-product
        # The bias term is not added when we are using the the feature similarity because it wouldn't matter
        # MNLI uses three types of labels but hans and mnli2 uses two types only.
        
        faiss_index = faiss_utils.FAISSIndex(2*(768+1), "Flat") #, metric=metric)
            
        if trained_on_task_name == "mnli" and train_task_name == "mnli":
            faiss_index = faiss_utils.FAISSIndex(3*(768+1), "Flat") #, metric=metric)
            faiss_index.load(constants.MNLI_FAISS_INDEX_PATH_PREDFEAT)
        elif trained_on_task_name == "mnli-2" and train_task_name == "mnli-2":
            faiss_index.load(constants.MNLI2_FAISS_INDEX_PATH_PREDFEAT)
        elif trained_on_task_name == "hans" and train_task_name == "hans":
            faiss_index.load(constants.HANS_FAISS_INDEX_PATH_PREDFEAT)
        elif trained_on_task_name == "mnli-2" and train_task_name == "hans":
            faiss_index.load(constants.MNLI2_HANS_FAISS_INDEX_PATH_PREDFEAT)
        elif trained_on_task_name == "hans" and train_task_name == "mnli-2":
            faiss_index.load(constants.HANS_MNLI2_FAISS_INDEX_PATH_PREDFEAT)
        else:
            faiss_index = None
    # End Thean
        

    return faiss_index


def select_s_test_config(
        trained_on_task_name: str,
        train_task_name: str,
        eval_task_name: str,
) -> Tuple[float, float, int]:

    if trained_on_task_name != train_task_name:
        # Only this setting is supported for now
        # basically, the config for this combination
        # of `trained_on_task_name` and `eval_task_name`
        # would be fine, so not raising issues here for now.
        if not all([
            trained_on_task_name == "mnli-2",
            train_task_name == "hans",
            eval_task_name == "hans"
        ]):
            raise ValueError("Unsupported as of now")

    if trained_on_task_name not in ["mnli", "mnli-2", "hans"]:
        raise ValueError

    if eval_task_name not in ["mnli", "mnli-2", "hans"]:
        raise ValueError

    # Other settings are not supported as of now
    if trained_on_task_name == "mnli" and eval_task_name == "mnli":
        s_test_damp = 5e-3
        s_test_scale = 1e4
        s_test_num_samples = 1000

    elif trained_on_task_name == "mnli-2" and eval_task_name == "mnli-2":
        s_test_damp = 5e-3
        s_test_scale = 1e4
        s_test_num_samples = 1000

    elif trained_on_task_name == "hans" and eval_task_name == "hans":
        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 2000

    elif trained_on_task_name == "mnli-2" and eval_task_name == "hans":
        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 1000

    elif trained_on_task_name == "hans" and eval_task_name == "mnli-2":
        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 2000

    else:
        raise ValueError

    return s_test_damp, s_test_scale, s_test_num_samples

# Thean ADD
# Called by:
#    visualization.main()
#    hans.one_experiment()
#    mnli.run_one_imitator_experiment()
# End Thean
def compute_influences_simplified(
        k: int,
        faiss_index: faiss_utils.FAISSIndex,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        train_dataset: torch.utils.data.DataLoader,
        use_parallel: bool,
        s_test_damp: float,                        # Thean: These are selected by select_s_test_config() CONSTANTS
        s_test_scale: float,                       # TThean: hese are selected by select_s_test_config() CONSTANTS
        s_test_num_samples: int,                   # Thean: NUmber of sampling. selected by select_s_test_config() CONSTANTS
        device_ids: Optional[List[int]] = None,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
        faiss_index_use_mean_features_as_query: bool = False,   # Thean: this is only true in hans.one_experiment()
        # Thean Start
        similarity: str = "feature",
        metric: str = "L2",
        direction: str = "similar"
        # Thean End
) -> Tuple[Dict[int, float]]:

    # Thean Add    
    if similarity not in ['feature', 'pred_feature']:
        raise ValueError("Choose similarity from `feature` or `pred_feature`")
    
    if metric not in ["L2", "inner_product", "cosine_similarity"]:
        raise ValueError("Choose metric from `L2`, `inner_product`, or `cosine_similarity`")
    
    if direction not in ["similar", "dissimilar", "mixed"]:
        raise ValueError("Choose direction from `similar`, `dissimilar`, or `mixed`")
    
    # Thean End
    
    # Make sure indices are sorted according to distances
    # KNN_distances[(
    #     KNN_indices.squeeze(axis=0)[
    #         np.argsort(KNN_distances.squeeze(axis=0))
    #     ] != KNN_indices)]

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    if faiss_index is not None:
        
        # Thean Modify: these are the original
        # features = misc_utils.compute_BERT_CLS_feature(model, **inputs)
        # features = features.cpu().detach().numpy()
        if similarity == "feature":
            features = misc_utils.compute_BERT_CLS_feature(model, **inputs)
            features = features.cpu().detach().numpy()
        elif similarity =="pred_feature":
            features = self_code.pred_feature_sim(model, inputs)
            if metric =="cosine_similarity":
                # Normalize the vector prior to searching
                faiss.normalize_L2(features)
        # End Thean Modify

        if faiss_index_use_mean_features_as_query is True:   # Thean: this is only true in hans.one_experiment()
            # We use the mean embedding as the final query here
            features = features.mean(axis=0, keepdims=True)
        
        # Thean Modify
        # Thean: KNN indices later go into nn_influence_utils.compute_influences()
        #KNN_distances, KNN_indices = faiss_index.search(  
        #    k=k, queries=features)
        
        #Thean Add
        if direction == "similar":
            _, KNN_indices = faiss_index.search(k=k, queries=features)
        elif direction == "dissimilar":
            features_rev = -1*features
            _, KNN_indices = faiss_index.search(k=k, queries=features_rev)
        elif direction == "mixed":
            middle_index = k//2 # floor division rounds down
            _, KNN_indices = faiss_index.search(k=middle_index, queries=features)
            
            # For the most dissimilar: https://github.com/facebookresearch/faiss/issues/1733
            features_rev = -1*features
            _, KNN_indices_rev = faiss_index.search(k=(k-middle_index), queries=features_rev)
            
            # convert to list for easier manipulation ; it originally looks like np.array([[3455,4903,...]])
            # Then concatenate list
            KNN_indices = KNN_indices[0] + KNN_indices_rev[0]
            KNN_indices.sort() # sort it ascendingly so that nn_influence_utils.compute_influences() runs faster
        else:
            raise ValueError("Choose direction from `similar`, `dissimilar` or `mixed`.")
        # End Thean
    else:
        KNN_indices = None

    if not use_parallel:
        model.cuda()
        # Thean Add: batch_train_data_loader -- batch 1 train_data RANDOM; used to estimate s_test
        batch_train_data_loader = misc_utils.get_dataloader(
            train_dataset,
            batch_size=1,
            random=True)
        # Thean Add: instance_train_data_loader -- batch 1 train_data FIX; used to calculate influence scores
        instance_train_data_loader = misc_utils.get_dataloader(
            train_dataset,
            batch_size=1,
            random=False)

        influences, _, _ = nn_influence_utils.compute_influences(
            n_gpu=1,
            device=torch.device("cuda"),
            batch_train_data_loader=batch_train_data_loader,
            instance_train_data_loader=instance_train_data_loader,
            model=model,
            test_inputs=inputs,
            params_filter=params_filter,
            weight_decay=constants.WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores,
            s_test_damp=s_test_damp,               # These are selected by select_s_test_config() CONSTANTS
            s_test_scale=s_test_scale,             # These are selected by select_s_test_config() CONSTANTS
            s_test_num_samples=s_test_num_samples, # J. These are selected by select_s_test_config() CONSTANTS
            train_indices_to_include=KNN_indices,  # Top K indices to be considered
            precomputed_s_test=precomputed_s_test) # Thean Add: default is None
    else:
        if device_ids is None:
            raise ValueError("`device_ids` cannot be None")

        influences, _ = parallel.compute_influences_parallel(
            # Avoid clash with main process
            device_ids=device_ids,
            train_dataset=train_dataset,
            batch_size=1,
            model=model,
            test_inputs=inputs,
            params_filter=params_filter,
            weight_decay=constants.WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores,
            s_test_damp=s_test_damp,
            s_test_scale=s_test_scale,
            s_test_num_samples=s_test_num_samples,
            train_indices_to_include=KNN_indices,
            return_s_test=False,
            debug=False)

    return influences

