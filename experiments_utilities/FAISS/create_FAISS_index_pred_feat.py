from experiments import hans
#HANS_MNLI2 = hans.create_FAISS_index_pred_feat(train_task_name="mnli-2", trained_on_task_name ="hans")
#HANS_MNLI2.save("faiss_index_pred_feat/HANS_MNLI2.index")

#MNLI2_HANS = hans.create_FAISS_index_pred_feat(train_task_name="hans", trained_on_task_name ="mnli-2")
#MNLI2_HANS.save("faiss_index_pred_feat/MNLI2_HANS.index")

MNLI = hans.create_FAISS_index_pred_feat(train_task_name="mnli", trained_on_task_name ="mnli")
MNLI.save("faiss_index_pred_feat/MNLI.index")

#MNLI2 = hans.create_FAISS_index_pred_feat(train_task_name="mnli-2", trained_on_task_name ="mnli-2")
#MNLI2.save("faiss_index_pred_feat/MNLI2.index")

#HANS = hans.create_FAISS_index_pred_feat(train_task_name="hans", trained_on_task_name ="hans")
#HANS.save("faiss_index_pred_feat/HANS.index")