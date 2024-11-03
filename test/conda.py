# torch installed using pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu12.5 to benefite from gpu
import os
import torch
import json
import torchaudio

from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from huggingface_hub import login, hf_hub_download

# pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16 --extra-index-url https://download.pytorch.org/whl/cu121
# pip install -U git+https://git@github.com/facebookresearch/audiocraft
# conda install ffmpeg

def require_environ(name):
    value = os.environ.get(name)
    if (value is not None):
        return value
    else:
        print("Missing required environ", name)
        exit(1)

def music_gen_small():
    print("Unimplemented !")
    exit(1)

def model_stable_audio_open():
    
    if (os.environ.get("OUTPUT_TEMPLATE") is not None):
        requirements = """{
    "options": {
        "MODEL": {
            "hidden": true,
            "kind": "String",
            "value": "stable_audio_open"
        },
        "OUTPUT_PATH": {
            "hidden": true,
            "kind": "String"
        },
        "HF_API_KEY": {
            "hidden": true,
            "kind": "String"
        },
        "CONDITION_PROMPT": {
            "kind": "String",
            "value": "140bpm dark techno aubuant sounds creepy"
        },
        "SECONDS_TOTAL": {
            "kind": "Float",
            "value": 30,
            "range": {
                "start": 0,
                "end": 30
            }
        },
        "STEPS": {
            "kind": "Int",
            "value": 100,
            "range": {
                "start": 1,
                "end": 200
            }
        },
        "CFG_SCALE": {
            "kind": "Int",
            "value": 3,
            "range": {
                "start": 1,
                "end": 10
            }
        },
        "CFG_SCALE": {
            "kind": "Float",
            "value": 3,
            "range": {
                "start": 1,
                "end": 10
            }
        },
        "SIGMA_MIN": {
            "kind": "Float",
            "value": 3,
            "range": {
                "start": 0,
                "end": 1000
            }
        },
        "SIGMA_MAX": {
            "kind": "Float",
            "value": 500,
            "range": {
                "start": 0,
                "end": 1000
            }
        }
    }
}"""
        print(requirements)
        exit(0)
    
    output_path     = require_environ("OUTPUT_PATH")
    hf_api_key      = require_environ("HF_API_KEY")
    prompt          = require_environ("CONDITION_PROMPT")
    seconds_total   = require_environ("SECONDS_TOTAL")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    login(token=hf_api_key)
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    model = model.to(device)
    

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0, 
        "seconds_total": int(seconds_total)
    }]

    # Generate stereo audio
    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=3,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save(output_path, output, sample_rate)

model_switch = {
    "stable_audio_open": model_stable_audio_open,
    "music_gen_small": music_gen_small
}

name = require_environ("MODEL")

model_switch[name]()