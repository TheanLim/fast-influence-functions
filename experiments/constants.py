# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

WEIGHT_DECAY = 0.005

MNLI_MODEL_PATH = "tuned_param/mnli-10ep"
HANS_MODEL_PATH = "tuned_param/hans-10ep"
MNLI2_MODEL_PATH = "tuned_param/mnli-2label-10ep"
MNLI_IMITATOR_MODEL_PATH = "tuned_param/mnli-imitator-10ep"

#### Original Feature similarities only
# Trained and used in MNLI
MNLI_FAISS_INDEX_PATH = "faiss_index/MNLI.index"
# Trained and used in HANS
HANS_FAISS_INDEX_PATH = "faiss_index/HANS.index"
# Trained and used in MNLI-2
MNLI2_FAISS_INDEX_PATH = "faiss_index/MNLI2.index"
# Trained on MNLI2 and used in HANS
MNLI2_HANS_FAISS_INDEX_PATH = "faiss_index/MNLI2_HANS.index"
# Trained on HANS and used in MNLI2
HANS_MNLI2_FAISS_INDEX_PATH = "faiss_index/HANS_MNLI2.index"
####

#### Thean Add
# Trained and used in MNLI
MNLI_FAISS_INDEX_PATH_PREDFEAT = "faiss_index_pred_feat/MNLI.index"
# Trained and used in HANS
HANS_FAISS_INDEX_PATH_PREDFEAT = "faiss_index_pred_feat/HANS.index"
# Trained and used in MNLI-2
MNLI2_FAISS_INDEX_PATH_PREDFEAT = "faiss_index_pred_feat/MNLI2.index"
# Trained on MNLI2 and used in HANS
MNLI2_HANS_FAISS_INDEX_PATH_PREDFEAT = "faiss_index_pred_feat/MNLI2_HANS.index"
# Trained on HANS and used in MNLI2
HANS_MNLI2_FAISS_INDEX_PATH_PREDFEAT = "faiss_index_pred_feat/HANS_MNLI2.index"
#### Thean End

# Thean not sure about this
MNLI_TRAIN_INPUT_COLLECTIONS_PATH = "experiments_outputs/imitator/train_inputs_collections.pth"

HANS_DATA_DIR = "export/home/Data/HANS"
GLUE_DATA_DIR = "export/home/Data/Glue/MNLI"
MNLI_TRAIN_FILE_NAME = "export/home/Data/Glue/MNLI/train.tsv"
MNLI_EVAL_MATCHED_FILE_NAME = "export/home/Data/Glue/MNLI/test_matched.tsv"
MNLI_EVAL_MISMATCHED_FILE_NAME = "export/home/Data/Glue/MNLI/test_mismatched.tsv"
HANS_TRAIN_FILE_NAME = "export/home/Data/HANS/heuristics_train_set.txt"
HANS_EVAL_FILE_NAME = "export/home/Data/HANS/heuristics_evaluation_set.txt"

# These are wrong.
# Experiments specific
MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR = "mnli_retrain_influence/1"
MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR2 = "mnli_retrain_influence/2"

# Some useful default hparams for influence functions
DEFAULT_INFLUENCE_HPARAMS = {
    # `train_on_task_name`
    "mnli": {
        # `eval_task_name`
        "mnli": {
            "damp": 5e-3,
            "scale": 1e4,
            "num_samples": 1000
        }
    }
}
