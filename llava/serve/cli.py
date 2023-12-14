import argparse
import torch
import requests
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import random
import string

def get_string(string_length):
    return ''.join(random.choice(string.ascii_lowercase[0:2]) for _ in range(string_length))

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()
    assert not (args.image_file and args.video_file)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                     args.load_8bit, args.load_4bit, device=args.device)
    # print(model, tokenizer, processor)
    image_processor = processor['image']
    video_processor = processor['video']
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    image = args.image_file
    video = args.video_file
    # print(image, video)
    if args.image_file:
        
        
        ###################### CODE FOR IMAGE DOWNLOAD#########
        # error_image.jpg
        image_file = args.image_file
        if (image_file.startswith('http')):
            img_format = '.jpg' 
            if 'format=png' in image_file:
                img_format = '.png'
            try:
                response = requests.get(image_file)
                response.raise_for_status()  
                image = get_string(16) + img_format
                # image = 'image'+img_format

                with open(image, 'wb') as f:
                    f.write(response.content)
                    print("write success")
            except Exception as e:
                image = 'error_image.jpg'
                print(f"Error downloading image from {args.image_file}: {str(e)} \nUsing error_image.jpg")
        ###########################################################
        
        
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        if type(image_tensor) is list:
            tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            tensor = image_tensor.to(model.device, dtype=torch.float16)
        key = ['image']
        # print(tensor.shape)
    elif args.video_file:
        
        ########### CODE FOR VIDEO DOWNLOAD ##############

        # error_video.mp4
        video_file = args.video_file
        video = 'error_video.mp4'
        if video_file.startswith('http'):
            try:
                # Download the video
                response = requests.get(video_file)
                response.raise_for_status()  

                # video_file = 'video.mp4'
                video_file = get_string(15)
                video = video_file
                with open(video, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Error downloading video from {args.video_file}: {str(e)} \nUsing error_video.mp4")
        ########################################################
        
        video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
        if type(video_tensor) is list:
            tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)
        key = ['video']
        # print(tensor.shape)
    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            inp = DEFAULT_X_TOKEN['IMAGE'] + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        elif video is not None:
            # first message
            inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            video = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if args.image_file:
            input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['IMAGE'], return_tensors='pt').unsqueeze(0).cuda()
        elif args.video_file:
            input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()
        # print(input_ids.shape)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[tensor, key],
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--video-file", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)
