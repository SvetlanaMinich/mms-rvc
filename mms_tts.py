from transformers import VitsModel, AutoTokenizer
from infer_rvc_python import BaseLoader
import torch
import os
import numpy as np

models_dir = 'models'


class MmsModels:
    def __init__(self, devices=['cuda:0']) -> None:
        self.models_dir = 'models'
        # Load models and tokenizers for each device
        self.models = {}
        self.tokenizers = {}
        self.converters = {}
        self.converter_tag = "marasov"

        for device in devices:
            if not torch.cuda.is_available() and device.startswith("cuda"):
                print(f"{device} is not available. Skipping...")
                continue
            
            # Initialize and configure the converter for each device
            self.converters[device] = BaseLoader(only_cpu=False, hubert_path='hubert_base.pt', rmvpe_path='rmvpe.pt')
            self.converters[device].apply_conf(
                tag="marasov",
                file_model="voice_models/masarov/MaraSov_e600_s3600.pth",
                pitch_algo="rmvpe+",
                pitch_lvl=0,
                file_index="voice_models/masarov/added_IVF290_Flat_nprobe_1_MaraSov_v2.index",
                index_influence=0.66,
                respiration_median_filtering=3,
                envelope_ratio=0.25,
                consonant_breath_protection=0.33,
                device=device  # Use the current device for the converter
            )

            # Load models
            self.models[device] = {
                "urd": VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic", cache_dir=self.models_dir).to(device).eval(),
                "eng": VitsModel.from_pretrained("facebook/mms-tts-eng", cache_dir=self.models_dir).to(device).eval(),
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
                "eng": AutoTokenizer.from_pretrained("facebook/mms-tts-eng", cache_dir=self.models_dir),
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

            self.tts_eng("Preloading the model once, as you've shown", device=device)

    
    def voice_conv(self, out_array, device):
        waveform_int16 = np.int16(out_array * 32767)
        result_array, sample_rate = self.converters[device].generate_from_cache(
            audio_data=(waveform_int16, 16_000),
            tag=self.converter_tag,
        )
        return result_array, sample_rate

    def tts(self, text, lang, device):
        if lang not in self.tokenizers[device]:
            raise ValueError(f"Language '{lang}' is not supported on device '{device}'.")

        inputs = self.tokenizers[device][lang](text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.models[device][lang](**inputs.to(device))
        return self.voice_conv(outputs.waveform[0].cpu().float().numpy(), device)

    def tts_urd(self, text, device):
        return self.tts(text, "urd", device)

    def tts_eng(self, text, device):
        return self.tts(text, "eng", device)

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