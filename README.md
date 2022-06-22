# FastIF with improved search heuristic
This is a work in progress to replicate and improve FastIF search heuristic.
<br />
Codes are heavily annotated to facilitate understandings.
[Link to the Original Paper](https://arxiv.org/abs/2012.15781)

## Recall of kNN
**Replication of the original method:**
![recall_feat_L2](figs/experiment_plots_copy/recall@m/recall@m_feature_similar_L2.svg)
<br />
**Improved Heuristic:**
![recall_predFeat_IP](figs/experiment_plots_copy/recall@m/recall@m_pred_feature_dis_n_similar_InnerProduct.svg)

## Explainability of Influential Examples
**Replication of the original method:**
![imitator_feat_L2](figs/experiment_plots_copy/imitator/Loss_of_the_Imitator_Model_Feature_Similarity_L2.svg)
<br />
**Improved Heuristic:**
![imitator_predFeat_IP](figs/experiment_plots_copy/imitator/Loss_of_the_Imitator_Model_Feature_Predictive_Similarity_IP.svg)

## Error Correction
### MultiNLI→ HANS→ HANS
**Replication of the original method:**
![error_correction_hans_feat_L2](figs/experiment_plots_copy/hans_augmentation/Hans_Augmentation(HANS)_Feature_Similarity_L2.svg)
<br />
**Improved Heuristic:**
![error_correction_hans_predFeat_IP](figs/experiment_plots_copy/hans_augmentation/Hans_Augmentation(HANS)_Feature_Predictive_Similarity_InnerProduct.svg)
### MultiNLI→ MultiNLI→ HANS
**Replication of the original method:**
![error_correction_mnli_feat_L2](figs/experiment_plots_copy/hans_augmentation/Hans_Augmentation(MNLI)_Feature_Similarity_L2.svg)
<br />
**Improved Heuristic:**
![error_correction_mnli_predFeat_IP](figs/experiment_plots_copy/hans_augmentation/Hans_Augmentation(MNLI)_Feature_Predictive_Similarity_InnerProduct.svg)



# Requirements
Please see `requirements.txt` for detailed dependencies. The major ones include
- `python 3.6 or later` (for type annotations and f-string)
- `pytorch==1.5.1`
- `transformers==3.0.2`
