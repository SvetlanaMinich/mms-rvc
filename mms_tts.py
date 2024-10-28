from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
import os
import numpy as np
from scipy.signal import resample

from styletts2 import tts
import time
import nltk
nltk.download('punkt_tab')

models_dir = 'models'


class MmsModels:
    def __init__(self, devices=['cuda:0']) -> None:
        self.models_dir = 'models'
        # Load models and tokenizers for each device
        self.models = {}
        self.tokenizers = {}

        for device in devices:
            if not torch.cuda.is_available() and device.startswith("cuda"):
                print(f"{device} is not available. Skipping...")
                continue

            # Load models
            self.models[device] = {
                "urd": VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic", cache_dir=self.models_dir).to(device).eval(),
                "eng": tts.StyleTTS2(model_checkpoint_path='libritts/epochs_2nd_00020.pth', config_path='libritts/config.yml'),
                "ara": VitsModel.from_pretrained("facebook/mms-tts-ara", cache_dir=self.models_dir).to(device).eval(),
                "deu": VitsModel.from_pretrained("facebook/mms-tts-deu", cache_dir=self.models_dir).to(device).eval(),
                "rus": VitsModel.from_pretrained("facebook/mms-tts-rus", cache_dir=self.models_dir).to(device).eval(),
                "hin": VitsModel.from_pretrained("facebook/mms-tts-hin", cache_dir=self.models_dir).to(device).eval(),
                "ben": VitsModel.from_pretrained("facebook/mms-tts-ben", cache_dir=self.models_dir).to(device).eval(),
                "mya": VitsModel.from_pretrained("facebook/mms-tts-mya", cache_dir=self.models_dir).to(device).eval(),
                "nld": VitsModel.from_pretrained("facebook/mms-tts-nld", cache_dir=self.models_dir).to(device).eval(),
                "fin": VitsModel.from_pretrained("facebook/mms-tts-fin", cache_dir=self.models_dir).to(device).eval(),
                "heb": VitsModel.from_pretrained("facebook/mms-tts-heb", cache_dir=self.models_dir).to(device).eval(),
                "ind": VitsModel.from_pretrained("facebook/mms-tts-ind", cache_dir=self.models_dir).to(device).eval(),
                "spa": VitsModel.from_pretrained("facebook/mms-tts-spa", cache_dir=self.models_dir).to(device).eval(),
                "kan": VitsModel.from_pretrained("facebook/mms-tts-kan", cache_dir=self.models_dir).to(device).eval(),
            }
            
            # Load tokenizers
            self.tokenizers[device] = {
                "urd": AutoTokenizer.from_pretrained("facebook/mms-tts-urd-script_arabic", cache_dir=self.models_dir),
                "ara": AutoTokenizer.from_pretrained("facebook/mms-tts-ara", cache_dir=self.models_dir),
                "deu": AutoTokenizer.from_pretrained("facebook/mms-tts-deu", cache_dir=self.models_dir),
                "rus": AutoTokenizer.from_pretrained("facebook/mms-tts-rus", cache_dir=self.models_dir),
                "hin": AutoTokenizer.from_pretrained("facebook/mms-tts-hin", cache_dir=self.models_dir),
                "ben": AutoTokenizer.from_pretrained("facebook/mms-tts-ben", cache_dir=self.models_dir),
                "mya": AutoTokenizer.from_pretrained("facebook/mms-tts-mya", cache_dir=self.models_dir),
                "nld": AutoTokenizer.from_pretrained("facebook/mms-tts-nld", cache_dir=self.models_dir),
                "fin": AutoTokenizer.from_pretrained("facebook/mms-tts-fin", cache_dir=self.models_dir),
                "heb": AutoTokenizer.from_pretrained("facebook/mms-tts-heb", cache_dir=self.models_dir),
                "ind": AutoTokenizer.from_pretrained("facebook/mms-tts-ind", cache_dir=self.models_dir),
                "spa": AutoTokenizer.from_pretrained("facebook/mms-tts-spa", cache_dir=self.models_dir),
                "kan": AutoTokenizer.from_pretrained("facebook/mms-tts-kan", cache_dir=self.models_dir),
            }
            

    def tts(self, text, lang, device):
        inputs = self.tokenizers[device][lang](text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.models[device][lang](**inputs.to(device))
        return outputs.waveform[0].cpu().float().numpy()

    def tts_urd(self, text, device):
        return self.tts(text, "urd", device)

    def tts_eng(self, text, device):
        output = self.models[device]["eng"].inference(text, target_voice_path="def_male.wav")
        num_samples = int(len(output) * (16000 / 24000))  # Calculate the new number of samples
        output_resampled = resample(output, num_samples)
        return output_resampled

    def tts_ara(self, text, device):
        return self.tts(text, "ara", device)

    def tts_deu(self, text, device):
        return self.tts(text, "deu", device)

    def tts_rus(self, text, device):
        return self.tts(text, "rus", device)

    def tts_hin(self, text, device):
        return self.tts(text, "hin", device)

    def tts_ben(self, text, device):
        return self.tts(text, "ben", device)

    def tts_mya(self, text, device):
        return self.tts(text, "mya", device)

    def tts_nld(self, text, device):
        return self.tts(text, "nld", device)

    def tts_fin(self, text, device):
        return self.tts(text, "fin", device)

    def tts_heb(self, text, device):
        return self.tts(text, "heb", device)

    def tts_ind(self, text, device):
        return self.tts(text, "ind", device)

    def tts_spa(self, text, device):
        return self.tts(text, "spa", device)

    def tts_kan(self, text, device):
        return self.tts(text, "kan", device)