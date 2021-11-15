# FastIF with improved search heuristic
This is a work in progress to replicate and improve FastIF search heuristic.
Codes are heavily annotated to facilitate understandings.
[Link to the Original Paper](https://arxiv.org/abs/2012.15781)

## Recall of kNN
**Replication of the original method:**
![recall_predFeat_IP](figs/experiment_plots_copy/recall@m/recall@m_feature_similar_L2.png)
<br />
**Improved Heuristic:**
![recall_predFeat_IP](figs/experiment_plots_copy/recall@m/recall@m_pred_feature_dis_n_similar_InnerProduct.png)

## Explainability of Influential Examples
**Replication of the original method:**
![imitator_predFeat_IP](figs/experiment_plots_copy/imitator/Loss of the Imitator Mode - Feature Similarity (L2).png)
<br />
**Improved Heuristic:**
![imitator_predFeat_IP](figs/experiment_plots_copy/imitator/Loss of the Imitator Mode - Feature+Predictive Similarity (InnerProduct).png)



# Requirements
Please see `requirements.txt` for detailed dependencies. The major ones include
- `python 3.6 or later` (for type annotations and f-string)
- `pytorch==1.5.1`
- `transformers==3.0.2`
