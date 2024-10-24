from transformers import VitsModel, AutoTokenizer
import torch

from scipy.io.wavfile import write
import numpy as np
from TTS.api import TTS

models_dir = 'models'


class MmsModels:
    def __init__(self, devices=['cuda:0']) -> None:
        self.models_dir = 'models'
        self.models = {}
        self.tokenizers = {}
        self.vc = {}

        # self.converter_tag = "masarov"

        # self.converter = BaseLoader(only_cpu=False, hubert_path='hubert_base.pt', rmvpe_path='rmvpe.pt')
        # self.converter.apply_conf(
        #         tag="masarov",
        #         file_model="voice_models/masarov/MaraSov_e600_s3600.pth",
        #         pitch_algo="rmvpe+",
        #         pitch_lvl=0,
        #         file_index="voice_models/masarov/added_IVF290_Flat_nprobe_1_MaraSov_v2.index",
        #         index_influence=0.66,
        #         respiration_median_filtering=3,
        #         envelope_ratio=0.25,
        #         consonant_breath_protection=0.33
        #     )

        for device in devices:
            if not torch.cuda.is_available() and device.startswith("cuda"):
                print(f"{device} is not available. Skipping...")
                continue
            
            self.vc[device] = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24").to(device)
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

    
    # def voice_conv(self, out_array, device=None):
    #     waveform_int16 = np.int16(out_array * 32767)
    #     result_array, sample_rate = self.converter.generate_from_cache(
    #         audio_data=(waveform_int16, 16_000),
    #         tag=self.converter_tag,
    #     )
    #     return result_array, sample_rate

    def voice_conversion(self, out, device):
        scaled_waveform = np.int16(out / np.max(np.abs(out)) * 32767)
        sample_rate = 16000
        write("output.wav", sample_rate, scaled_waveform)
        wav = self.vc[device].voice_conversion(source_wav="output.wav",
                                               target_wav='def_male.wav')
        return wav

    def tts(self, text, lang, device):
        inputs = self.tokenizers[device][lang](text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.models[device][lang](**inputs.to(device))
        return self.voice_conversion(out=outputs.waveform[0].cpu().float().numpy(),
                                     device=device)

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