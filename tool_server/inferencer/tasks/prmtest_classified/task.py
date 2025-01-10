
from mr_eval.utils.task_utils import *
from mr_eval.utils.utils import *
from mr_eval.utils.log_utils import get_logger
import os

logger = get_logger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "config.yaml"
file_path = os.path.join(current_dir, file_name)
task_config = load_task_config(file_path)
if "dataset_path" in task_config and os.path.isabs(task_config["dataset_path"]) == False:
    task_config["dataset_path"] = os.path.join(current_dir,task_config["dataset_path"])

def load_data_function():
    
    raw_data = load_dir_of_jsonl_data_function_default(task_config)
    meta_data = []
    correct_sample_classification = "redundency"
    filter_dict = {}
    for idx, item in enumerate(raw_data):
        item_idx = item["idx"]
        question = item["modified_question"]
        steps = item["modified_process"]
        error_steps = item["error_steps"]
        classification = item["classification"]

        if classification == correct_sample_classification:
            correct_idx = f"correct_{item_idx}"
            if filter_dict.get(correct_idx) is None:
                correct_sample = dict(
                    idx=correct_idx,
                    question=item["original_question"],
                    steps=item["original_process"],
                    error_steps=[],
                    classification="correct"
                )
                meta_data.append(correct_sample)
                filter_dict[correct_idx] = 1
        classification_idx = f"{classification}_{item_idx}"
        if filter_dict.get(classification_idx) is None:
            res = dict(idx=classification_idx,question=question,steps=steps,error_steps=error_steps,classification=classification)
            meta_data.append(res)
            filter_dict[classification_idx] = 1
    
    ## Show statistics
    classification_dict = {}
    for item in meta_data:
        classification = item["classification"]
        classification_dict[classification] = classification_dict.get(classification,0)+1 
        
    for k,v in classification_dict.items():
        logger.info(f"Classification: {k}, number: {v}")
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


def evaluate_function(results,meta_data):
    meta_data_dict = {meta["idx"]: meta for meta in meta_data}
    classification_types = set([meta["classification"] for meta in meta_data])
    metric_types = ["correct_step_acc","wrong_step_acc","total_step_acc","first_error_acc"]
    halucination_specified_dict = {}
    total_metric_lists = {}
    for metric in metric_types+["similarity"]:
        halucination_specified_dict[metric] = {i:[] for i in classification_types}
        total_metric_lists[metric] = []
    halucination_specified_dict["f1_matrix"] = {i:dict(TP=0,FP=0,TN=0,FN=0) for i in classification_types}
    total_metric_lists["f1_matrix"] = dict(TP=0,FP=0,TN=0,FN=0)
    

    detailed_logs = []
    valid_num = 0
    total_num = len(meta_data)
    
    ## Filter out repeated items
    filtered_dict = {}
    filtered_results = []
    for result in results:
        idx = result["idx"]
        if filtered_dict.get(idx) is None and meta_data_dict.get(idx) is not None:
            filtered_dict[idx] = 1
            filtered_results.append(result)
    
    assert abs(len(filtered_results) - len(meta_data)) < 5, f"filtered_results number: {len(filtered_results)}, meta_data number: {len(meta_data)}"

    correct_ids_dict = {meta["idx"]:1 for meta in meta_data if meta["classification"] == "correct"} 
    correct_predictions  = [prediction for prediction in filtered_results if prediction["idx"] in correct_ids_dict]
    other_predictions = [prediction for prediction in filtered_results if prediction["idx"] not in correct_ids_dict]
    correct_model_response_acc_dict = {}
    
    ## First evaluate the correct samples
    for prediction in correct_predictions:
        idx = prediction["idx"]
        reference_item = meta_data_dict[idx]
        error_steps = reference_item["error_steps"]    
        classifcation = reference_item["classification"]
        assert classifcation == "correct"
        
        if "validity" in prediction and not prediction["validity"]:
            log = dict(
                idx=idx,
                error_steps=error_steps,
                classifcation=classifcation,
                prediction=None,
                results=None,
                )
        else:
            labels = prediction["scores"]["step_level_validity_labels"] if "step_level_validity_labels" in prediction["scores"] else None
            res_dict = eval_on_hallucination_step(error_steps,labels)
                
            for metric in metric_types:
                # total_metric_lists[metric].extend(res_dict[f'{metric}_list'])
                halucination_specified_dict[metric][classifcation].extend(res_dict[f'{metric}_list'])
            halucination_specified_dict["f1_matrix"][classifcation]["TP"] += res_dict["f1_matrix"]["TP"]
            halucination_specified_dict["f1_matrix"][classifcation]["FP"] += res_dict["f1_matrix"]["FP"]
            halucination_specified_dict["f1_matrix"][classifcation]["TN"] += res_dict["f1_matrix"]["TN"]
            halucination_specified_dict["f1_matrix"][classifcation]["FN"] += res_dict["f1_matrix"]["FN"]
            
            correct_model_response_acc_dict[idx] = res_dict["model_response_acc"]   
            log = dict(
                idx=idx,
                error_steps=error_steps,
                classifcation=classifcation,
                prediction=prediction,
                results=res_dict,
            )
            detailed_logs.append(log)
    
    ## Then evaluate the other sample types
    for prediction in other_predictions:
        idx = prediction["idx"]
        
        if "validity" in prediction and not prediction["validity"]:
            log = dict(
                idx=idx,
                hallucination_steps=None,
                hallucination_types=None,
                prediction=None,
                results=None,
                validitiy=False,
                )
        else:
            valid_num += 1
            try:
                reference_item = meta_data_dict[idx]
            except:
                logger.info(f"idx {idx} not found in meta_data_dict")
                continue
            error_steps = reference_item["error_steps"]
            classifcation = reference_item["classification"]
            
            if (classifcation == "redundency" or classifcation == "circular") and "step_level_redundancy_labels" in prediction["scores"]:
                labels = prediction["scores"]["step_level_redundancy_labels"]
                labels = [ not i for i in labels]
                res_dict = eval_on_hallucination_step(error_steps,labels,redundency_label=False)
            else:
                labels = prediction["scores"]["step_level_validity_labels"] if "step_level_validity_labels" in prediction["scores"] else None
                res_dict = eval_on_hallucination_step(error_steps,labels,redundency_label=False)
                
            for metric in metric_types:
                total_metric_lists[metric].extend(res_dict[f'{metric}_list'])
                halucination_specified_dict[metric][classifcation].extend(res_dict[f'{metric}_list'])
            halucination_specified_dict["f1_matrix"][classifcation]["TP"] += res_dict["f1_matrix"]["TP"]
            halucination_specified_dict["f1_matrix"][classifcation]["FP"] += res_dict["f1_matrix"]["FP"]
            halucination_specified_dict["f1_matrix"][classifcation]["TN"] += res_dict["f1_matrix"]["TN"]
            halucination_specified_dict["f1_matrix"][classifcation]["FN"] += res_dict["f1_matrix"]["FN"]
            
            total_metric_lists["f1_matrix"]["TP"] += res_dict["f1_matrix"]["TP"]
            total_metric_lists["f1_matrix"]["FP"] += res_dict["f1_matrix"]["FP"]
            total_metric_lists["f1_matrix"]["TN"] += res_dict["f1_matrix"]["TN"]
            total_metric_lists["f1_matrix"]["FN"] += res_dict["f1_matrix"]["FN"]
            
            correct_idx = "correct_"+idx[len(f"{classifcation}_"):]
            correct_item_acc = correct_model_response_acc_dict.get(correct_idx)
            item_acc = res_dict["model_response_acc"]
            if correct_item_acc and item_acc != -1:
                abs_similarity = abs(item_acc - correct_item_acc)  
                total_metric_lists["similarity"].append(abs_similarity)
                halucination_specified_dict["similarity"][classifcation].append(abs_similarity)
            log = dict(
                        idx=idx,
                        error_steps=error_steps,
                        classifcation=classifcation,
                        prediction=prediction,
                        results=res_dict,
                    )
        detailed_logs.append(log)
    
    
    ## Calculate final results
    total_final_results = {metric:sum(total_metric_lists[metric])/len(total_metric_lists[metric]) if len(total_metric_lists[metric])>0 else -1 for metric in metric_types+["similarity"]}
    hallucination_type_final_results = {metric:{k:sum(v)/len(v) if len(v)>0 else -1 for k,v in halucination_specified_dict[metric].items()} for metric in metric_types+["similarity"]}
    validitiy_rate = valid_num / total_num
    
    
    ## Calculate F1 score
    TP = total_metric_lists["f1_matrix"]["TP"]
    FP = total_metric_lists["f1_matrix"]["FP"]
    FN = total_metric_lists["f1_matrix"]["FN"]
    TN = total_metric_lists["f1_matrix"]["TN"]
    total_precision = TP / (TP + FP) if (TP + FP) != 0 else -1
    total_recall = TP / (TP + FN) if (TP + FN) != 0 else -1
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) != 0 else -1
    negative_precision = TN / (TN + FN) if (TN + FN) != 0 else -1
    negative_recall = TN / (TN + FP) if (TN + FP) != 0 else -1
    negative_f1 = 2 * negative_precision * negative_recall / (negative_precision + negative_recall) if (negative_precision + negative_recall) != 0 else -1
    total_final_results["precision"] = total_precision
    total_final_results["recall"] = total_recall
    total_final_results["f1"] = total_f1
    total_final_results["negative_precision"] = negative_precision
    total_final_results["negative_recall"] = negative_recall
    total_final_results["negative_f1"] = negative_f1
    
    for metric in ["precision","recall","f1","negative_precision","negative_recall","negative_f1"]:
        hallucination_type_final_results[metric] = {}
    for classification in classification_types:
        TP = halucination_specified_dict["f1_matrix"][classification]["TP"]
        FP = halucination_specified_dict["f1_matrix"][classification]["FP"]
        FN = halucination_specified_dict["f1_matrix"][classification]["FN"]
        TN = halucination_specified_dict["f1_matrix"][classification]["TN"]
        precision = TP / (TP + FP) if (TP + FP) != 0 else -1
        recall = TP / (TP + FN) if (TP + FN) != 0 else -1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else -1
        negative_precision = TN / (TN + FN) if (TN + FN) != 0 else -1
        negative_recall = TN / (TN + FP) if (TN + FP) != 0 else -1
        negative_f1 = 2 * negative_precision * negative_recall / (negative_precision + negative_recall) if (negative_precision + negative_recall) != 0 else -1
        hallucination_type_final_results["precision"][classification] = precision
        hallucination_type_final_results["recall"][classification] = recall
        hallucination_type_final_results["f1"][classification] = f1
        hallucination_type_final_results["negative_precision"][classification] = negative_precision
        hallucination_type_final_results["negative_recall"][classification] = negative_recall
        hallucination_type_final_results["negative_f1"][classification] = negative_f1
    
    res = dict(
        total_hallucination_results=total_final_results,
        hallucination_type_results=hallucination_type_final_results,
        validitiy_rate=validitiy_rate,
        detailed_logs=detailed_logs,
    )
    return res




def eval_on_hallucination_step(hallucination_steps, labels, redundency_label=False):
    ## Important: hallucination_steps are 0-indexed
    hallucination_steps = [i-1 for i in hallucination_steps]
    ## Important: hallucination_steps are 0-indexed
    if redundency_label:
        POSITIVE_LABEL = 0
        NEGATIVE_LABEL = 1
    else:
        POSITIVE_LABEL = 1
        NEGATIVE_LABEL = 0

    correct_step_acc = []
    wrong_step_acc = []
    total_step_acc = []
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    first_error_location = min(hallucination_steps) if len(hallucination_steps)>0 else -1
    first_error_acc = None
    for idx in range(len(labels)):
        
        if idx == first_error_location:
            if labels[idx] == NEGATIVE_LABEL:
                first_error_acc = 1
            else:
                first_error_acc = 0
        
        if idx in hallucination_steps:
            if labels[idx] == POSITIVE_LABEL:
                wrong_step_acc.append(0)
                total_step_acc.append(0)
                FP += 1
            else:
                wrong_step_acc.append(1)
                total_step_acc.append(1)
                TN += 1
        else:
            if labels[idx] == POSITIVE_LABEL:
                correct_step_acc.append(1)
                total_step_acc.append(1)
                TP += 1
            else:
                correct_step_acc.append(0)
                total_step_acc.append(0)
                FN += 1
                
    
    correct_step_acc_value = sum(correct_step_acc)/len(correct_step_acc) if len(correct_step_acc)>0 else -1
    wrong_step_acc_value = sum(wrong_step_acc)/len(wrong_step_acc) if len(wrong_step_acc)>0 else -1
    total_step_acc_value = sum(total_step_acc)/len(total_step_acc) if len(total_step_acc)>0 else -1
    model_response_acc = sum(labels)/len(labels) if len(labels)>0 else -1
    
    return dict(
        correct_step_acc=correct_step_acc_value,
        wrong_step_acc=wrong_step_acc_value,
        total_step_acc=total_step_acc_value,
        first_error_acc=first_error_acc,
        model_response_acc=model_response_acc,
        f1_matrix = dict(TP=TP,FP=FP,TN=TN,FN=FN),
        
        
        correct_step_acc_list=correct_step_acc,
        wrong_step_acc_list=wrong_step_acc,
        total_step_acc_list=total_step_acc,
        first_error_acc_list=[first_error_acc] if first_error_acc is not None else [],
        model_response_acc_list=[model_response_acc] if model_response_acc != -1 else [],
    )