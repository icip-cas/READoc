from scoring import TextEvaluate, MathEvaluate, HeadEvaluate, TableEvaluate, OrderEvaluate
from segmentation import extract_materials
from standardization import clean_md
import os
from glob import glob
import fire
import json

def file_evaluate(gold_md, pred_md):
    gold_json_content = extract_materials(clean_md(gold_md))
    pred_json_content = extract_materials(clean_md(pred_md))

    overall_result = TextEvaluate(gold_json_content['overall'], pred_json_content['overall'])
    plain_result = TextEvaluate(gold_json_content['subtask']['plain'], pred_json_content['subtask']['plain'])
    math_result = MathEvaluate(gold_json_content['subtask']['math'], pred_json_content['subtask']['math'])
    head_result = HeadEvaluate(gold_json_content['subtask']['heads'], pred_json_content['subtask']['heads'])
    table_result = TableEvaluate(gold_json_content['subtask']['table'], pred_json_content['subtask']['table'])
    order_result = OrderEvaluate(gold_json_content['overall'], pred_json_content['overall'], gold_json_content['segments'], pred_json_content['segments'])

    return {'overall': overall_result, 'plain': plain_result, 'math': math_result, 'head': head_result, 'table': table_result, 'order': order_result}


def e2e_evaluate(gold_dir, pred_dir, result_json):

    all_results = dict()
    avg_results = dict()
    gold_file_list = glob(f"{gold_dir}/*.md")

    for gold_md in gold_file_list:
        with open(gold_md, 'r') as f:
            gold_md_content = f.read()
        
        pred_md = os.path.join(pred_dir, gold_md.split('/')[-1])

        try:
            with open(pred_md, 'r') as f:
                pred_md_content = f.read()
        except:
            pred_md_content = ""
        
        mmd_result = file_evaluate(gold_md_content, pred_md_content)

        for type in mmd_result:
            if type not in all_results.keys():
                all_results[type] = dict()
            for metric_type in mmd_result[type].metric_dict:
                if metric_type not in all_results[type]:
                    all_results[type][metric_type] = dict()
                for specific_metric in mmd_result[type].metric_dict[metric_type]:
                    if specific_metric not in all_results[type][metric_type]:
                        all_results[type][metric_type][specific_metric] = []
                    all_results[type][metric_type][specific_metric].append(mmd_result[type].metric_dict[metric_type][specific_metric])

    for type in all_results:
        avg_results[type] = dict()
        for sub_key in all_results[type]:
            avg_results[type][sub_key] = dict()
            for sub_sub_key in all_results[type][sub_key]:
                avg_results[type][sub_key][sub_sub_key] = sum(all_results[type][sub_key][sub_sub_key]) / len(all_results[type][sub_key][sub_sub_key])

    try: # arXiv
        result_dict = {
            'text_eds': avg_results['plain']['metrics']['edit_dist_sim'],
            'text_f1': avg_results['plain']['metrics']['f_measure'],
            'head_eds': avg_results['head']['concated']['edit_dist_sim'],
            'head_teds': avg_results['head']['logical']['teds'],
            'inline_eds': avg_results['math']['inline_concated']['edit_dist_sim'],
            'outline_eds': avg_results['math']['outline_concated']['edit_dist_sim'],
            'table_eds': avg_results['table']['concated']['edit_dist_sim'],
            'table_teds': avg_results['table']['mapped']['teds_min'],
            'seg_kt': avg_results['order']['segment']['kendall_tau'],
            'word_kt': avg_results['order']['word']['kendall_tau'],
            'seg_sp': avg_results['order']['segment']['spearmanr'],
            'word_sp': avg_results['order']['word']['spearmanr'],
            }
    except: # GitHub
        result_dict = {
            'text_eds': avg_results['plain']['metrics']['edit_dist_sim'],
            'text_f1': avg_results['plain']['metrics']['f_measure'],
            'head_eds': avg_results['head']['concated']['edit_dist_sim'],
            'head_teds': avg_results['head']['logical']['teds'],
            'seg_kt': avg_results['order']['segment']['kendall_tau'],
            'word_kt': avg_results['order']['word']['kendall_tau'],
            'seg_sp': avg_results['order']['segment']['spearmanr'],
            'word_sp': avg_results['order']['word']['spearmanr'],
            }         
    
    with open(result_json, 'w') as f:
        json.dump(result_dict, f)
     

if __name__ == "__main__":
    fire.Fire(e2e_evaluate)
