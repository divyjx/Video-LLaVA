
<div align= "center">
    <h1> 🔥 Finetuned VideoLLaVA</h1>
</div>

## Overview

The model we used for this project is Video-LLaVA. It is a multimodal LLM with capabilites of interpreting images and videos. To do behaviour and content simulation we finetuned this model over dataset given here : https://huggingface.co/datasets/divyjx/VideoLLaVA_dataset . This enabled the model to take text and media content of a tweet and make predictions on number of likes (behaviour simulation) or take amount of likes and media content and given text content to get those amount of likes (content simuation). To check that finetuing does provide a better result we compare it with results from a model finetuned on just 10k datapoints.
<ul>
    <li> Model finetuned on 10k datapoints : https://huggingface.co/divyjx/VideoLLaVA10k
    <li> Final model : https://huggingface.co/divyjx/VideoLLaVA
</ul>

## Usage

Follow these steps to obtain answers for the questions:

1. Make a conda environment and clone this repository
   ```python
    %%shell
    eval "$(conda shell.bash hook)"
    git clone https://github.com/PKU-YuanGroup/Video-LLaVA
    cd Video-LLaVA
    conda create -n videollava python=3.10 -y
    conda activate videollava
   
2. Clone this repository
3. Download the dependencies needed using :
   ```python
    %%shell
    eval "$(conda shell.bash hook)"
    conda activate videollava
    pip install --upgrade pip  # enable PEP 660 support
    pip install -e .
    pip install -e ".[train]"
    pip install flash-attn --no-build-isolation
    pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
    pip install tensorboardX
    pip install pycocoevalcap
   ```
4. With current directory set at Video-LLaVA , download our finetuned model using :
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id= "divyjx/VideoLLaVA", local_dir = 'checkpoints/Video-LLaVA-7B', local_dir_use_symlinks=True)
   ```
5. With current directory set at Video-LLaVA , download our dataset using :
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id= "divyjx/VideoLLaVA_dataset", local_dir = 'tuning_data', local_dir_use_symlinks=True, repo_type = 'dataset')
   ```
6. With that you have all the neccessary files to run inference and evaluation


## How to prepare dataset ?
Refer to preprocess_1.ipynb and preprocess_2.ipynb for preparing data. 

## How to do fine-tuning ?
Refer to VideoLlaVa_train.ipynb for fine-tuning.
Alternatively, you can run the following command after setting up the enviornment and preprocessing data. Remember to customize the finetune_final.sh script based on your device specifications. Ensure that the current directory is set to Video-LLaVA . 
```python
%%shell
eval "$(conda shell.bash hook)"
conda activate videollava
bash scripts/v1_5/finetune_final.sh
```
## How to do inference ?

- With current directory set as Video-LLaVA run the following command :
  ```python
    %%shell
    eval "$(conda shell.bash hook)"
    conda activate videollava
    python inference.py --model-path "checkpoints/Video-LLaVA-7B" --instruct-path "tuning_data/your_data.json" --output-file-path "your_model_output.json" --    
    video-    folder "videos" --image-folder "images
  ```

## How to do evaluation ?

- With current directory set as Video-LLaVA run the following command :
  ```python
    %%shell
    eval "$(conda shell.bash hook)"
    conda activate videollava
    python eval.py --prediction-file-path "your_model_output.json" --input-file-path "tuning_data/your_data.json" --type-of-data "your_type"
  ```
  Here your_type should be either behaviour or content

## Notebooks

Alternatively you can try running VideoLLaVA-inference.ipynb for inference and evaluation.
Remember to connect to GPU when running inference or evaluation.


## 🛠️ Requirements and Installation
* Python >= 3.10
* Pytorch == 2.0.1
* CUDA Version >= 11.7

## 👍 Acknowledgement
* [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) The model we finetuned upon.
* (https://arxiv.org/abs/2309.00359) The paper we refered for generating prompts for finetuning.



