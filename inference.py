import torch
import os
import json
import argparse
import requests
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def main(args):
############### LOAD MODEL #################
    disable_torch_init()
    model_path = args.model_path
    instruct_path = args.instruct_path
    output_file_path = args.output_file_path
    with open(instruct_path, 'r') as instruct_data_json:
        instruct_data = json.load(instruct_data_json)
    device = args.device
    load_4bit, load_8bit = args.load_4bit, args.load_8bit
    conv_mode = args.conv_mode
    model_name = get_model_name_from_path(model_path)
    image_folder = args.image_folder
    video_folder = args.video_folder
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
################## LOOP FOR BATCH INFERENCE ############
    for entry in instruct_data:
        if 'id' in entry and (('video' in entry) or ('image' in entry)) and 'conversations' in entry:
            typee = entry["conversations"][0]["value"][1:6]
            prompt = entry["conversations"][0]["value"]
            id = entry['id']
            conv = conv_templates[conv_mode].copy()
            roles = conv.roles
            ###################IMAGE######################
            if typee == 'image':
              image_path = entry['video']
              image, error = image_download(image_folder, image_path)
              image_processor = processor['image']

              image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
              if type(image_tensor) is list:
                  tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
              else:
                  tensor = image_tensor.to(model.device, dtype=torch.float16)
              key = ['image']

              #print(f"{roles[1]}: {prompt}")
              conv.append_message(conv.roles[0], prompt)
              conv.append_message(conv.roles[1], None)
              prompt = conv.get_prompt()
              input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['IMAGE'], return_tensors='pt').unsqueeze(0).cuda()
              stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
              keywords = [stop_str]
              stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

              with torch.inference_mode():
                  output_ids = model.generate(
                      input_ids,
                      images=[tensor, key],
                      do_sample=True,
                      temperature=0.2,
                      max_new_tokens=1024,
                      use_cache=True,
                      stopping_criteria=[stopping_criteria])

              outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
              #############ADD OUTPUTS TO A JSON FILE###############
              new_data = {
                'id': f'{id}',
                'error': f'{error}',              #Check if we used error image or video
                'type': f'{typee}',
                'output': f'{outputs}'
              }
              with open(output_file_path, 'a+') as json_file:
                json_file.seek(0)
                json.dump(new_data, json_file, indent=2)

            #####################VIDEO#########################
            elif typee == 'video':
              video_path = entry['video']
              video, error = video_download(video_folder, video_path)
              video_processor = processor['video']
              video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
              if type(video_tensor) is list:
                  tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
              else:
                  tensor = video_tensor.to(model.device, dtype=torch.float16)
              key = ['video']

              #print(f"{roles[1]}: {prompt}")
              conv.append_message(conv.roles[0], prompt)
              conv.append_message(conv.roles[1], None)
              prompt = conv.get_prompt()
              input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()
              stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
              keywords = [stop_str]
              stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

              with torch.inference_mode():
                  output_ids = model.generate(
                      input_ids,
                      images=[tensor, key],
                      do_sample=True,
                      temperature=0.1,
                      max_new_tokens=1024,
                      use_cache=True,
                      stopping_criteria=[stopping_criteria])

              outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
              #############ADD OUTPUTS TO A JSON FILE###############
              new_data = {
                'id': f'{id}',
                'error': f'{error}',
                'type': f'{typee}',
                'output': f'{outputs}'
              }
              with open(output_file_path, 'a+') as json_file:
                json_file.seek(0)
                json.dump(new_data, json_file, indent=2)

def image_download(image_folder, image_file):
  error = False
  if (image_file.startswith('http')):
      img_format = '.jpg'
      if 'format=png' in image_file:
          img_format = '.png'
      try:
          response = requests.get(image_file)
          response.raise_for_status()  # Check if the request was successful
          image_path = 'image'+img_format
          image = os.path.join(image_folder, image_path)
          with open(image, 'wb') as f:
              f.write(response.content)
      except Exception as e:
          image = os.path.join(image_folder, 'error_image.jpg')
          error = True
          print(f"Error downloading image from {image_file}: {str(e)} \nUsing error_image.jpg")
  return image, error


def video_download(video_folder, video_file):
  error = False
  if video_file.startswith('http'):
      try:
          # Download the video
          response = requests.get(video_file)
          response.raise_for_status()

          video_path = 'video.mp4'
          video = os.path.join(video_folder, video_path)
          with open(video, 'wb') as f:
              f.write(response.content)
      except Exception as e:
          video = os.path.join(video_folder, 'error_video.mp4')
          error = True
          print(f"Error downloading video from {video_file}: {str(e)} \nUsing error_video.mp4")
  return video, error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="'LanguageBind/Video-LLaVA-7B'")
    parser.add_argument("--instruct-path", type=str, default = "'behaviour_100.json'")
    parser.add_argument("--output-file-path", type=str, default = "'output.json'")
    parser.add_argument("--prediction-type", type=str, default = "'behaviour'")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--load-8bit", type=bool, default = False)
    parser.add_argument("--load-4bit", type=bool, default = True)
    parser.add_argument("--video-folder", type=str, default = None)
    parser.add_argument("--image-folder", type=str, default = None)
    args = parser.parse_args()
    main(args)