# torch installed using pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu12.5 to benefite from gpu
import os
import json
import socket
import struct
import time
import re

import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


sock = None

difussion_models = {
    "MusicGenSmall": {
        "options": {
            "MODEL": { "hidden": True, "kind": "String", "value": "MusicGenSmall" },
            "POINT": { "hidden": True, "kind": "String", "value": "facebook/musicgen-small" },
            "CONDITION_PROMPT": { "kind": "String", "value": "140bpm dark techno hypnotic drums loop" },
            "SECONDS_TOTAL": { "kind": "Float", "value": 20, "range": { "start": 0, "end": 30 } },
        }
    },
    "MusicGenMedium": {
        "options": {
            "MODEL": { "hidden": True, "kind": "String", "value": "MusicGenMedium" },
            "POINT": { "hidden": True, "kind": "String", "value": "facebook/musicgen-medium" },
            "CONDITION_PROMPT": { "kind": "String", "value": "140bpm dark techno hypnotic drums loop" },
            "SECONDS_TOTAL": { "kind": "Float", "value": 20, "range": { "start": 0, "end": 30 } },
        }
    },
    "MusicGenLarge": {
        "options": {
            "MODEL": { "hidden": True, "kind": "String", "value": "MusicGenLarge" },
            "POINT": { "hidden": True, "kind": "String", "value": "facebook/musicgen-large" },
            "CONDITION_PROMPT": { "kind": "String", "value": "140bpm dark techno hypnotic drums loop" },
            "SECONDS_TOTAL": { "kind": "Float", "value": 20, "range": { "start": 0, "end": 30 } },
        }
    },
    "MusicGenMelody": {
        "options": {
            "MODEL": { "hidden": True, "kind": "String", "value": "MusicGenMelody" },
            "POINT": { "hidden": True, "kind": "String", "value": "facebook/musicgen-melody" },
            "CONDITION_PROMPT": { "kind": "String", "value": "140bpm dark techno hypnotic drums loop" },
            "CONDITION_SAMPLE": { "kind": "String", "value": "C:\\Users\\corbe\\Music\\2024-07-30_08-01-32.wav" },
            "SECONDS_TOTAL": { "kind": "Float", "value": 20, "range": { "start": 0, "end": 30 } },
        }
    },
    "MusicGenSongStarter": {
        "options": {
            "MODEL": { "hidden": True, "kind": "String", "value": "MusicGenSongStarter" },
            "POINT": { "hidden": True, "kind": "String", "value": "nateraw/musicgen-songstarter-v0.2" },
            "CONDITION_PROMPT": { "kind": "String", "value": "140bpm dark techno hypnotic drums loop" },
            "CONDITION_SAMPLE": { "kind": "String", "value": "C:\\Users\\corbe\\Music\\2024-07-30_08-01-32.wav" },
            "SECONDS_TOTAL": { "kind": "Float", "value": 20, "range": { "start": 0, "end": 30 } },
        }
    }
}

def generate_filename(prompt, idx):
    truncated_string = prompt[:12]
    pattern = r'[<>:"/\\|?*\x00-\x1F]'
    sanitized_string = re.sub(pattern, '_', truncated_string)
    sanitized_string = sanitized_string.strip()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{truncated_string}_{idx}_{timestamp}"
    return filename

def get_available_diffusion_model(request):
    return [
        "MusicGenSmall",
        "MusicGenMedium",
        "MusicGenLarge",
        "MusicGenMelody",
        "MusicGenSongStarter"
    ]

def get_diffusion_model_template_task(request):
    return difussion_models[request['model']]

def run_diffusion_model_template_task(request):
    options = request['template']['options']
    point = options['POINT']['value']
    output_directory = options['OUTPUT_DIRECTORY']['value']
    descriptions = [options['CONDITION_PROMPT']['value']]

    model = MusicGen.get_pretrained(point)
    model.set_generation_params(duration=8)
    wav = model.generate(descriptions)
    
    output = []
    for idx, one_wav in enumerate(wav):
        file_path = f'{output_directory}\\{generate_filename(descriptions[0], idx)}'
        audio_write(file_path, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        output.append(f'{file_path}.wav')
    
    return {
        'assets': output
    }

remote_procedure = {
    "GetDiffusionModelTemplateTask": get_diffusion_model_template_task,
    "RunDiffusionModelTemplateTask": run_diffusion_model_template_task,
    "GetAvailableDiffusionModel": get_available_diffusion_model,
}

def handle(message):
    if message['id'] == "Call":
        try:
            payload = remote_procedure[message['procedure_id']](message['payload'])
            send({"id": "CallBack", "call_id": message['call_id'], "payload": payload})
        except Exception as e:
            log(f"remote procedure error: {e}")
            send({"id": "CallBack", "call_id": message['call_id'], "payload": None})
    else:
        log(f"unknown packet: {message['id']}")

def send(packet):
    packet_content = json.dumps(packet)
    packed_int = struct.pack('!i', len(packet_content))
    string_bytes = packet_content.encode('utf-8')
    buffer = packed_int + string_bytes
    sock.sendall(buffer)
    
def log(message, level = "info"):
    send({"id":"Log", "message": message, "level": level})

def require_environ(name):
    value = os.environ.get(name)
    if (value is not None):
        return value
    else:
        print("Missing required environ", name)
        exit(1)

try:
    print("connecting ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", int(require_environ("PORT"))))
    print("Connected !")
    while True:
        packet_length_data = sock.recv(4)
        if len(packet_length_data) < 4:
            print("Connection closed by client")
            break
        packet_length = struct.unpack('!i', packet_length_data)[0]
        packet_data = sock.recv(packet_length)
        if len(packet_data) < packet_length:
            print("Incomplete packet data")
            break
        raw = packet_data.decode('utf-8')
        message = json.loads(raw);        
        to_send = send({"id": "Ack", "request": message})
        try:
            handle(message)
        except Exception as e:
            print(e)
            log(f"failed to handle request {message}: {e}", level="Error")

except Exception as e:
    print(f"Error: {e}")
finally:
    sock.close()
