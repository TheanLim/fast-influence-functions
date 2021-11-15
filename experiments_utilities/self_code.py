from typing import List, Dict, Tuple, Optional, Union, Any
from experiments import influence_helpers
from experiments import misc_utils
from experiments import constants
from tqdm import tqdm
import transformers
from transformers import TrainingArguments
import faiss
import os
import numpy as np
import numpy
import glob
import torch
import copy
import matplotlib.pyplot as plt
import matplotlib
from influence_utils import faiss_utils


#####################################################
#import glob
#import torch

def recall_main(ms = [10,100, 1000],
                ks = [10, 100, 1000, 5000, 10000, 50000, 100000],
                correct_pattern = 'experiments_outputs/KNN-recall.only-correct.3.1000.*',
                incorrect_pattern = 'experiments_outputs/KNN-recall.only-incorrect.3.1000.*',
                similarity = "feature",
                metric: str = "L2",
                mixed_direction: bool = False,
                figname: str = ""
               )-> None:
    '''
    Calculate recall@m and visualize it.
    Currently supports mnli-mnli but not others such as mnli-2 or hans.
    
    ms: Number of influence points
    ks: Numbers of Nearest Neighbors
    in/correct_pattern: patterns to be used to search for files containing influence_values 
                        of each training data to each testing data.
    similarity: `feature` or `pred_feature` (prediction*feature) similarities to be used in the KNN search
    metric: Distance metric to use in KNN search. {`L2`, `inner_product`, `cosine_similarity`}
    mixed_direction: Search for the most SIMILAR neighbor when calculate recall for 'Most Helpful' cases;
                     Search for the most DISSIMILAR neighbor when calculate recall for 'Most Helpful' cases;
                     Applicable to `inner_product` and `cosine_similarity` only.
    figname: Name for the plots
    
    This function first reads and concatenates files containing influence_values using patterns given.
    Then, it calculates the recall score for cases where the model predicted test cases correctly, and 
    saves the calculated outputs as a file.
    The same steps are repeated for cases where the model predicted test cases INcorrectly.
    It visualizes the recall score by m and k after that.
    '''
    
    ########### Assertions
    if similarity not in ['feature', 'pred_feature']:
        raise ValueError("Choose similarity from `feature` or `pred_feature`")
    
    if metric not in ["L2", "inner_product", "cosine_similarity"]:
        raise ValueError("Choose metric from `L2`, `inner_product`, or `cosine_similarity`")
    
    if mixed_direction and metric not in ["inner_product", "cosine_similarity"]:
        raise ValueError("`mixed_direction` is only applicable for inner_product or cosine_similarity metrics")
    ############ End Assertions
    
    # For naming output files
    md = "dis_n_similar" if mixed_direction else "similar"
    distance = "L2" if metric == "L2" else "InnerProduct" if metric == "inner_product" else "CosineSimilarity"

    correct_output_collections={}
    incorrect_output_collections={}
    
    for name in glob.glob(correct_pattern):
        tmp = torch.load(name)
        correct_output_collections.update(tmp)
    correct_output_collections = dict(sorted(correct_output_collections.items())) 
    print("Total Correct Keys: ",len(correct_output_collections.keys()))
        
    for name in glob.glob(incorrect_pattern):
        tmp = torch.load(name)
        incorrect_output_collections.update(tmp)
    incorrect_output_collections = dict(sorted(incorrect_output_collections.items())) 
    print("Total Incorrect Keys: ",len(incorrect_output_collections.keys()))
    
    print("Recall - correct")
    recall_dict_correct = recall(ms,ks,correct_output_collections, 
                                 similarity = similarity, 
                                 metric = metric,
                                 mixed_direction = mixed_direction)
    correct_fn = checkfile("experiments_outputs/recall_dict_correct_"+similarity+"_"+md+"_"+metric+".pth")
    torch.save(recall_dict_correct, correct_fn)
    
    print("Recall - incorrect")
    recall_dict_incorrect = recall(ms,ks,incorrect_output_collections, 
                                   similarity= similarity, 
                                   metric = metric,
                                   mixed_direction = mixed_direction)
    incorrect_fn = checkfile("experiments_outputs/recall_dict_incorrect_"+similarity+"_"+md+"_"+metric+".pth")
    torch.save(recall_dict_incorrect, incorrect_fn)
    
    if figname == "":
        figname = "recall@m_"+similarity+"_"+md+"_"+distance+".png"
    
    if similarity =="feature":
        super_title = "Recall@m using Feature Similarity ("+distance+")"
    else:
        super_title = "Recall@m using Prediction and Feature Similarities ("+distance+")"
        
    recall_m_viz(recall_dict_correct, recall_dict_incorrect, ks, ms, 
                 super_title = super_title,
                 figname = figname)
######################################################
# from experiments import influence_helpers
# from experiments import misc_utils
# from experiments import constants
# from tqdm import tqdm
# import transformers
# from transformers import TrainingArguments
# import faiss

def recall(ms:List[int],
           ks:List[int], 
           output_collections: Dict[int, Dict[str, Any]],
           similarity: str = "feature",
           metric: str = "L2",
           mixed_direction: bool = False
          ) -> Dict[int, Dict[int, Any]]:
    '''
    Calculate recall@m.
    Currently supports mnli-mnli but not others such as mnli-2 or hans.
    
    ms: Number of influence points
    ks: Numbers of Nearest Neighbors
    output_collections: Data containing influence_values of each training data to each testing data.
                        See its data strucuture below.
    similarity: `feature` or `pred_feature` (prediction*feature) similarities to be used in the KNN search
    metric: Distance metric to use in KNN search. {`L2`, `inner_product`, `cosine_similarity`}
    mixed_direction: Search for the most SIMILAR neighbor when calculate recall for 'Most Helpful' cases;
                     Search for the most DISSIMILAR neighbor when calculate recall for 'Most Helpful' cases;
                     Applicable to `inner_product` and `cosine_similarity` only.
                     
    outputs_collections = {
        test_index:{"test_index": test_index,
                    "influences": {train_index:influence_val},
                    "s_test": s_test,
                    "time": timer.elapsed,
                    "correct": prediction_is_correct},
        ...
    }
    
    Returns:
    {m:{k:([],[],[])}}  : A nested dictionary. [],[],[] represent lists of for MostHelpful, MostHarmful and 
                          MostInfluential recall scores across all testing indices presented in outputs_collections.
                          These lists are packed into a tuple for each m,k combination
    
    '''
    #### Assertions
    if similarity not in ['feature', 'pred_feature']:
        raise ValueError("Choose similarity from `feature` or `pred_feature`")
    
    if metric not in ["L2", "inner_product", "cosine_similarity"]:
        raise ValueError("Choose metric from `L2`, `inner_product`, or `cosine_similarity`")
    
    if mixed_direction and metric not in ["inner_product", "cosine_similarity"]:
        raise ValueError("`mixed_direction` is only applicable for inner_product or cosine_similarity metrics")
    ##### End Assertions
    
    
    tokenizer, model = misc_utils.create_tokenizer_and_model(
        constants.MNLI_MODEL_PATH)
    
    (mnli_train_dataset,
     mnli_eval_dataset) = misc_utils.create_datasets(
        task_name="mnli",
        tokenizer=tokenizer)

    faiss_index = influence_helpers.load_faiss_index(
        trained_on_task_name="mnli",
        train_task_name="mnli",
        similarity = similarity)
    
    eval_instance_data_loader = misc_utils.get_dataloader(
        dataset=mnli_eval_dataset,
        batch_size=1,
        random=False)
    
    if similarity == "pred_feature":
        trainer = transformers.Trainer(
            model=model,
            args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100),
        )
    
    model.cuda()
    device = model.device
    
    # Dict to store recall by m and k
    mk_dict = {m:{k:([],[],[]) for k in ks} for m in ms}
    
    for test_index, outputs in tqdm(output_collections.items()):
            
        # top m
        influences = outputs["influences"] 
        #Thean ^: influences={train_index:influence_val} aka influence val of each train index to the test index
        
        # helpful and harmful
        try:
            helpful_indices, harmful_indices = misc_utils.get_helpful_harmful_indices_from_influences_dict(influences, n = max(ms))
        except ValueError as error:
            print(repr(error) +" at test_index "+str(test_index))
            continue
        # top influential s
        sorted_abs_indices = [key for key,_ in sorted(influences.items(), key=lambda item: abs(item[1]), reverse = True)]
        influential_indices = sorted_abs_indices # alias
        
        # top_k
        for eval_index, eval_inputs in enumerate(eval_instance_data_loader):
            if eval_index == test_index:
                test_inputs = eval_inputs
                break
        
        for k, v in test_inputs.items():
            test_inputs[k] = v.to(device)
        
        # This replace its followings - not tested
        features = input_to_features(model, test_inputs, trainer, similarity, metric)
        '''
        if similarity == "feature":
            features = misc_utils.compute_BERT_CLS_feature(model, **test_inputs)
            features = features.cpu().detach().numpy()
        elif similarity =="pred_feature":
            features = pred_feature_sim(model, test_inputs, trainer)
            if metric =="cosine_similarity":
                # Normalize the vector prior to searching
                faiss.normalize_L2(features)
        '''
                
        # Thean: the KNN_indices is equivalent to train_indices
        _, KNN_indices = faiss_index.search(k=max(ks), queries=features)
        if mixed_direction:
            # For the most dissimilar: https://github.com/facebookresearch/faiss/issues/1733
            features_rev = -1*features
            _, KNN_indices_rev = faiss_index.search(k=max(ks), queries=features_rev)

        # convert to list for easier manipulation ; it originally looks like np.array([[3455,4903,...]]) 
        KNN_indices = KNN_indices[0]
        if mixed_direction:
            KNN_indices_rev = KNN_indices_rev[0]
            
        # Calculate recalls
        for m in ms:
            
            # Ground Truth
            top_m_help = helpful_indices[:m]
            top_m_harm = harmful_indices[:m]
            top_m_influential = influential_indices[:m]
            
            for k in ks:
                top_k = KNN_indices[:k]

                if mixed_direction:
                    top_k_rev = KNN_indices_rev[:k]
                    
                    # Mixs half top_k and top_k_rev
                    top_k_mixed = [] 
                    middle_index = k//2 # floor division rounds down
                    top_k_mixed.extend(top_k[:middle_index])
                    top_k_mixed.extend(top_k_rev[:(k-middle_index)])
                    
                    helpful_score = len(intersection(top_k, top_m_help))/m
                    harmful_score = len(intersection(top_k_rev, top_m_harm))/m
                    influential_score = len(intersection(top_k_mixed, top_m_influential))/m
                else:
                    helpful_score = len(intersection(top_k, top_m_help))/m
                    harmful_score = len(intersection(top_k, top_m_harm))/m
                    influential_score = len(intersection(top_k, top_m_influential))/m

                recall_helpful, recall_harmful, recall_influential = mk_dict[m][k]
                recall_helpful.append(helpful_score)
                recall_harmful.append(harmful_score)
                recall_influential.append(influential_score)
  
                mk_dict[m][k] = recall_helpful, recall_harmful, recall_influential
    
        #break
    return mk_dict
######################################################
# import copy
# import numpy as np
# import matplotlib.pyplot as plt

def recall_m_viz(dict_correct: Dict[int, Dict[int, Any]],
                 dict_incorrect: Dict[int, Dict[int, Any]], 
                 ks: List = [10, 100, 1000, 5000, 10000, 50000, 100000], 
                 ms: List = [10, 100, 1000],
                 addtl_pt: List = [500000, 1.0, 0.0],
                 super_title: str = "Recall@m",
                 figname: str = ""
                )->None:
    '''
    addtl_pt: an ending point added assuming we run using k= 5x10^5, which is
              more than the entire eval dataset
    '''
    ######### Aesthetics ########
    alpha_map = {
        10:0.3,
        100:0.6,
        1000:1
    }
    color_map = {
        "correct": "darkcyan",
        "incorrect": "lightsalmon"
    }
    ######### End Aesthetics ########
    
    recall_dicts = {"correct":dict_correct,
                    "incorrect":dict_incorrect}
    titles=["Helpful", "Harmful", "Influential"]
    figname = "recall@m.png" if figname == "" else figname
    
    # Setting up plots config
    fig, ax = plt.subplots(1,3,sharey=True, dpi=200)
    fig.set_figwidth(20)
    #fig.set_figheight(8) 
    
    for i in range(3):
        for m in ms:
            a = alpha_map[m]
            for mode in ["correct", "incorrect"]:
                x = copy.deepcopy(ks)
                color = color_map[mode]   
                y_mean = []
                y_err = []
                for k in ks:
                    tmp = recall_dicts[mode][m][k][i]
                    y_mean.append(np.mean(tmp))
                    y_err.append(np.std(tmp))
                
                x.append(addtl_pt[0])
                y_mean.append(addtl_pt[1])
                y_err.append(addtl_pt[2])
                
                ax[i].errorbar(x, 
                               y_mean,
                               yerr=y_err,
                               fmt='--o', 
                               label = "recall@"+str(m)+" ("+mode+")", 
                               alpha = a, 
                               c = color)

                ax[i].set_xlabel('# Nearest Neigbors')
                ax[i].set_ylabel('Recall')
                ax[i].set_title('Most '+titles[i])
                ax[i].set_xticks(ks)
                ax[i].legend(loc = "upper left")
                ax[i].set_xscale("log")
        
    plt.suptitle(super_title)
    plt.show()
    fig.savefig(figname)
######################################################
# from experiments import misc_utils
# import faiss

def input_to_features(model:torch.nn.Module,
                      inputs:Dict[str, torch.Tensor],
                      trainer:transformers.Trainer = None,
                      similarity: str = "feature",
                      metric: str = "L2"
                     )->numpy.ndarray:
    '''
    Takes in data inputs and convert them into features that are usable for faiss search.
    The output `features` are transform according to the chosen similarity and metric
    
    similarity: `feature` or `pred_feature` (prediction*feature) similarities to be used in the KNN search
    metric: Distance metric to use in KNN search. {`L2`, `inner_product`, `cosine_similarity`}
    '''
    #### Assertions
    if similarity not in ['feature', 'pred_feature']:
        raise ValueError("Choose similarity from `feature` or `pred_feature`")
    
    if metric not in ["L2", "inner_product", "cosine_similarity"]:
        raise ValueError("Choose metric from `L2`, `inner_product`, or `cosine_similarity`")
    ####
    
    if similarity == "feature":
        features = misc_utils.compute_BERT_CLS_feature(model, **inputs)
        features = features.cpu().detach().numpy()
    elif similarity =="pred_feature":
        features = pred_feature_sim(model, inputs, trainer)
        if metric =="cosine_similarity":
            # Normalize the vector prior to searching
            faiss.normalize_L2(features)
    
    return features
######################################################

def pred_feature_sim(model:torch.nn.Module, 
                     inputs:Dict[str, torch.Tensor], 
                     trainer:transformers.Trainer = None
                    )->numpy.ndarray:
    '''
    Calculate the predictive * feature similarities
     = (pred_logits - one_hot_y) * features
     
    It also reshape the output becoming one vector per data instance,
    with dtype = float32. This is to make it compatible to FAISS index configs.
    
    Output:
    pred_features_vec: numpy.ndarray, shape[batch_size, x], dtype = float32
                       where x = num_labels*feature_len(768+1)
    '''
    if trainer is None:
        trainer = transformers.Trainer(
            model=model,
            args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100)
        )
    
    # **inputs is unpacking the inputs objects into multiple args
    features = misc_utils.compute_BERT_CLS_feature(model, **inputs)  
    features = features.cpu().detach().numpy()
    
    # features looks like np.array([[1,2,3], [4,5,6]])
    # Insert 1.0 at the front of each feature vector for Bias
    features_arr = []
    for i in range(len(features)):
        tmp = list(features[i])
        tmp.insert(0, 1.0)
        features_arr.append(tmp)
        
    del features  # prevent reference from happenning
    features = np.array(features_arr)
        
    preds, label_ids, step_eval_loss = misc_utils.predict(trainer=trainer,
                                                          model=model,
                                                          inputs=inputs)
    label_one_hot = one_hot(label_ids, trainer.model.num_labels)
    
    preds = softmax(preds)
    pred_label_diff = preds - label_one_hot
    
    # Outer Product with batches
    # Normal (nonbatch): np.einsum('i, j->ij', pred_label_diff, features)
    pred_features = np.einsum('bi, bj->bij', pred_label_diff, features)
    
    # Reshape the outer produc matrix so that each data has only one vec
    # The last batch_size may be less than 128 because of running out of data
    key = next(iter(inputs))
    batch_size = len(inputs[key]) # inputs[key] returns all the Ys(aka label_ids); its length is the batch size
    pred_features_vec = np.reshape(pred_features, (batch_size, -1))
    
    # Change type to be compatible to faiss
    pred_features_vec = pred_features_vec.astype('float32')
    
    return(pred_features_vec)
######################################################
# from experiments import constants
# from experiments import misc_utils
# import faiss
# import transformers
# from transformers import TrainingArguments
# from influence_utils import faiss_utils

def create_FAISS_index_sim_metrics(
    train_task_name: str,
    trained_on_task_name: str,
    similarity: str = "feature",
    metric: str = "L2"
) -> faiss_utils.FAISSIndex:
    '''
    Create Faiss indices using training data.
    The `similarity` and `metric` is used to configure the Faiss indices
    and possibly processing features.
    '''
    
    ######## Assertions ########
    if train_task_name not in ["mnli", "mnli-2", "hans"]:
        raise ValueError

    if trained_on_task_name not in ["mnli", "mnli-2", "hans"]:
        raise ValueError
    
    if similarity not in ['feature', 'pred_feature']:
        raise ValueError("Choose similarity from `feature` or `pred_feature`")
    
    if metric not in ["L2", "inner_product", "cosine_similarity"]:
        raise ValueError("Choose metric from `L2`, `inner_product`, or `cosine_similarity`")
     ######## End Assertions ########
    
    if trained_on_task_name == "mnli":
        # MNLI uses 3 labels for Y; The hans and mnli-2 uses two labels
        # so, MNLI can only pair with itself
        if train_task_name != "mnli":
            raise ValueError
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.MNLI_MODEL_PATH)
        
    if trained_on_task_name == "mnli-2":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.MNLI2_MODEL_PATH)

    if trained_on_task_name == "hans":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.HANS_MODEL_PATH)

    train_dataset, _ = misc_utils.create_datasets(
        task_name=train_task_name,
        tokenizer=tokenizer)
    
    model.cuda()
    device = model.device
    
    train_batch_data_loader = misc_utils.get_dataloader(
        dataset=train_dataset,
        batch_size=128, 
        random=False)
    
    trainer = transformers.Trainer(
        model=model,
        args=TrainingArguments(
        output_dir="./tmp-output",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=5e-5,
        logging_steps=100),
    )
    
    # 768 is the features length
    # For pred_feature: 768+1 because we add an additional "1" for the Bias term
    #                   multiply num_labels because of outer-product
    # The bias term is not added when we are using the the feature similarity because it wouldn't matter
    feature_num = trainer.model.num_labels*(768+1) if similarity == "pred_feature" else 768 
    faiss_index = faiss_utils.FAISSIndex(feature_num,"Flat", metric = metric)
    
    for inputs in tqdm(train_batch_data_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        
        if similarity == "feature":
            features = misc_utils.compute_BERT_CLS_feature(model, **test_inputs)
            features = features.cpu().detach().numpy()
        elif similarity =="pred_feature":
            features = pred_feature_sim(model, test_inputs, trainer)
            if metric =="cosine_similarity":
                # Normalize the vector prior to searching
                faiss.normalize_L2(features)
        
        faiss_index.add(pred_features_vec) 
        
    return faiss_index

#####################################################

#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot#:~:text=To%20place%20the%20legend%20outside,left%20corner%20of%20the%20legend.&text=A%20more%20versatile%20approach%20is,placed%2C%20using%20the%20bbox_to_anchor%20argument.
#import matplotlib.pyplot as plt
#from typing import List, Dict

def imitator_viz(
        Xs: List[float] = np.logspace(-5, -2.5, 50),
        Ys_dict: List[List[Dict[str, List[List[float]]]]] = None,
        ylabels:List[str] = ["Correct\n\nLoss", "Incorrect\n\nLoss"],
        title: str = "Loss of the Imitator Model"
) -> matplotlib.figure.Figure:
    
    if Ys_dict is None:
        raise ValueError("Please input Ys_dict")
    rs = len(Ys_dict)
    cs = len(Ys_dict[0])
    
    # Squeeze = false forces ax to always be 2dimensional
    fig, ax = plt.subplots(rs,cs,sharey=True, squeeze=False)#, dpi=150)
    fig.set_figwidth(40)
    #fig.set_figheight(8)

    color_map = {
        "random-neutral": "grey",
        "random-entailment": "salmon",
        "random-contradiction": "skyblue",
        "most-positive-influential": "darkred",
        "most-negative-influential": "steelblue"}
    
    for r in range(rs):
        for c in range(cs):
            y = Ys_dict[r][c]["losses"]
            for tag in y.keys():
                if tag not in color_map.keys():
                    raise ValueError(tag)

                color = color_map[tag]
                data = np.array(y[tag])
                is_random_data_point = "random" in tag
                
                if data.shape[0] != 1:
                    data_mean = data.mean(axis=0)
                    data_max = data.max(axis=0)
                    data_min = data.min(axis=0)
                    line = ax[r, c].plot(Xs, data_mean,
                             color=color,
                             linestyle=("--" if is_random_data_point else None),
                             label = tag+" (mean)" if r==0 else None)

                    fill = ax[r, c].fill_between(Xs, data_max, data_min,
                                     alpha=0.1,
                                     color=color,
                                     label = tag+ " (min/max)"if ((r == rs-1)or(rs<=1)) else None)

                else:
                    print("No Quality Check - might fail")
                    ax[r, c].plot(Xs, data[0, ...], color=color)
                ax[r, c].set_xscale("log")
                ax[r, c].set_yscale("log")
                ax[r, c].set_yticks([1, 0.01, 0.0001])
            
            # First Column
            if c == 0:
                ax[r, c].set_ylabel(ylabels[r])
            
            # Last Row
            if r == rs-1:
                ax[r, c].set_xlabel('Learning Rate')
            
            # More than a row: show line legend on the first, fill_between legend on the last
            if rs >1 :
                # First Row last column
                if c ==cs-1 and r ==0:
                    ax[r, c].legend(bbox_to_anchor=(1.04,0.0), loc="lower left", frameon = False)#, fontsize=5)
                # Last Row last column    
                if c ==cs-1 and r ==rs-1:
                    ax[r, c].legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon = False)#, fontsize=5)
            # Only one row
            else:
                # The last column
                if c == cs-1:
                    ax[r, c].legend(bbox_to_anchor=(1.04,0.5), loc="center left", frameon = False)#, fontsize=5)
                   
    plt.suptitle(title, fontsize=15)
    return fig

#####################################################
################## Utilities ########################
#####################################################
def intersection(lst1:List, lst2:List)->List: 
    '''
    Return a list with elements that exist in both
    list1 and list2
    '''
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

#####################################################
# import os
# Copied from https://stackoverflow.com/questions/29682971/auto-increment-file-name-python?rq=1

def checkfile(path:str)-> str:
    '''
    Auto Increment File Name
    Ex: checkfile("./Dockerfile") suggests me to use './Dockerfile_0' name instead.
    '''
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        return path

    root, ext = os.path.splitext(os.path.expanduser(path))
    dir       = os.path.dirname(root)
    fname     = os.path.basename(root)
    candidate = fname+ext
    index     = 0
    ls        = set(os.listdir(dir))
    while candidate in ls:
            candidate = "{}_{}{}".format(fname,index,ext)
            index    += 1
    return os.path.join(dir,candidate)

#######################################################
def between_str(string:str, substr1:str, substr2:str)-> str:
    '''
    Returns string that lies within two substrings
    Only able to detect the first occurence of substr
    
    Usage:
    string = "abckioki_10.pth"
    between_str(string, "_", ".pth")
    '''
    try:
        idx1 = string.index(substr1)
        idx2 = string.index(substr2)
    except ValueError:
        return ''
    
    return string[idx1 + len(substr1): idx2]

#######################################################
def after_str(string:str, substr1:str)-> str:
    '''
    Returns string that lies after a substring
    Only able to detect the first occurence of substr
    '''
    try:
        idx1 = string.index(substr1)
    except ValueError:
        return ''
    
    return string[idx1 + len(substr1):]

#######################################################
#import numpy as np

def softmax(x:numpy.ndarray)->numpy.ndarray:
    '''
    Calculate the softmax of x with its 
    dimension preserved. For example, for a 2D array
    it calculates the softmax with respect to each row.
    '''
    e = np.exp(x)
    out = e/np.sum(e,axis=-1,keepdims=True)
    return out

######################################################
#import numpy as np

def one_hot(arr:numpy.ndarray, max_num:int = None)->numpy.ndarray:
    '''
    Converts an array of integers into one-hot encoding format
    This only support 1D numpy array
    '''
    if max_num is None:
        max_num = np.max(arr)+1
    else:
        assert max_num>np.max(arr), "`max_num` has to be at least max(arr)+1"
        
    b = np.zeros((arr.size, max_num))
    b[np.arange(arr.size),arr] = 1
    return b

#####################################################
