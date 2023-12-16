import os
import json
import argparse
import re
import math
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def main(args):
###################### OPENING ALL JSON FILES ##########################
  prediction_file_path = args.prediction_file_path
  input_file_path = args.input_file_path
  type_of_data = args.type_of_data
  with open(input_file_path, 'r') as file:
    input_file = json.load(file)

  with open(prediction_file_path, 'r') as file:
    prediction_file = json.load(file)

####################### LOSS IN CASE OF BEHAVIOUR SIMULATION ########################
  if type_of_data == 'behaviour':
    #Merged data for calculation behaviour loss
    merged_data = []
    for item1 in input_file:
        for item2 in prediction_file:
            if item1['id'] == item2['id']:
                merged_item = {
                    'id': item1['id'],
                    'error': item2['error'],
                    'prediction': item2['output'],
                    'true_output': item1["conversations"][1]["value"]
                }
                merged_data.append(merged_item)
    loss = behaviour_loss(merged_data)
    print(f'RMSE Loss for {input_file_path} is : {loss}')

  ###################### LOSS IN CASE OF CONTENT SIMULATION ##########################
  elif type_of_data == 'content':
    transform_format_annotation(input_file_path, "annotations_for_content_loss.json")
    transform_format_results(prediction_file_path, "results_for_content_loss.json")
    annotation_file = 'annotations_for_content_loss.json'
    results_file = 'results_for_content_loss.json'
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)
    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
      print(f'{metric}: {score:.3f}')












################## FUNCTIONS TO CALCULATE BEHAVIOUR LOSS ##################
def behaviour_loss(merged_data):
    loss = []
    for entry in merged_data:
      id = entry['id']
      likes = entry['prediction']
      error = entry['error']
      true_likes = entry['true_output']
      if error == 'True':                                       # Don't calculate loss when error true
        continue
      if check_if_likes(likes):                                 # Check if output of LLM contains only likes 
        likes_int = int(extract_likes(likes))
        true_likes_int = int(extract_likes(true_likes))
        loss.append(rmse_loss(likes_int, true_likes_int))       # Add RMSE Loss
    return math.sqrt(sum(loss)/len(loss))


def rmse_loss(predicted, true):
    return (predicted - true) ** 2



################## CODE TO CHECK IF LIKES STRING CONTAIN ONLY INTERGERS AND TO EXTRACT THOSE INTEGERS #####################
def check_if_likes(input_string):
    pattern = r'^\d+</s>$'
    match = re.search(pattern, input_string)
    return bool(match)

def extract_likes(input_string):
  digits = re.findall(r'\d', input_string)
  result = ''.join(digits)
  return result














################### TRANSFORM FUNCTIONS OF COCO EVALUATION #########################

def transform_format_annotation(input_file, output_file):                     # Make annotation file(input file) match the format required for coco evaluation
    with open(input_file, 'r') as f:
        data = json.load(f)

    annotations = []
    annotations2 = []

    for item in data:
        annotation = {
            "image_id": int(item["id"]),
            "id": int(item["id"]),
            "caption": item["conversations"][1]["value"]
        }
        annotations.append(annotation)
    for item in data:
        annotation2 = {
            "id": int(item["id"]),
        }
        annotations2.append(annotation2)

    transformed_data = {
        "images" : annotations2,
        "annotations": annotations
        }

    with open(output_file, 'w') as f:
        json.dump(transformed_data, f, indent=2)
    return 0


def transform_format_results(input_file, output_file):                      # Make results file(prediction file) match the format required for coco evaluation
    with open(input_file, 'r') as f:
        data = json.load(f)

    annotations = []

    for item in data:
        annotation = {
            "image_id": int(item["id"]),
            "caption": item["output"]
        }
        annotations.append(annotation)

    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    return 0










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file-path", type=str, default = "'output.json'")      # Prediction File Path
    parser.add_argument("--input-file-path", type=str, default = "'behaviour_100.json'")    # Input File Path
    parser.add_argument("--type-of-data", type=str, default = "'behaviour'")                # Specify if evalution is to be done for "behaviour" or "content" simulation
    args = parser.parse_args()
    main(args)
