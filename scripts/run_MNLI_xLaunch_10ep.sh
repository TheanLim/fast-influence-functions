# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
export GLUE_DIR=export/home/Data/Glue
export OUT_DIR=tuned_param/mnli-10ep/

python run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name mnli \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/MNLI/ \
    --max_seq_length 128 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 10.0 \
    --output_dir $OUT_DIR \
    --weight_decay 0.005 \
    --save_steps 5000 \
    --logging_steps 100 \
    --save_total_limit 1
