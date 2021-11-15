from experiments import hans
MNLI = hans.create_FAISS_index(train_task_name="mnli", 
                               trained_on_task_name ="mnli", 
                               inner_product_metrics=True)
MNLI.save("faiss_index/MNLI.index")
