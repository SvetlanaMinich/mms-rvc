# langs:
# English            eng
# Urdu (script arab) urd
# Arabic             ara
# German             deu
# Russian            rus
# Hindi              hin

from transformers import VitsModel, AutoTokenizer
from infer_rvc_python import BaseLoader
import torch
import os

models_dir = 'models'


class MmsModels:
    def __init__(self) -> None:
        self.converter = BaseLoader(only_cpu=False, hubert_path='hubert_base.pt', rmvpe_path='rmvpe.pt')

        self.converter.apply_conf(
                tag="marasov",
                file_model="voice_models/masarov/MaraSov_e600_s3600.pth",
                pitch_algo="rmvpe+",
                pitch_lvl=0,
                file_index="voice_models/masarov/added_IVF290_Flat_nprobe_1_MaraSov_v2.index",
                index_influence=0.66,
                respiration_median_filtering=3,
                envelope_ratio=0.25,
                consonant_breath_protection=0.33
            )
        self.converter_tag = "marasov"

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        os.makedirs(models_dir, exist_ok=True)

        self.model_urd = VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic", cache_dir=models_dir)
        self.tokenizer_urd = AutoTokenizer.from_pretrained("facebook/mms-tts-urd-script_arabic", cache_dir=models_dir)
        self.model_urd.to(self.device).eval()

        self.model_eng = VitsModel.from_pretrained("facebook/mms-tts-eng", cache_dir=models_dir)
        self.tokenizer_eng = AutoTokenizer.from_pretrained("facebook/mms-tts-eng", cache_dir=models_dir)
        self.model_eng.to(self.device).eval()

        self.model_arab = VitsModel.from_pretrained("facebook/mms-tts-ara", cache_dir=models_dir)
        self.tokenizer_arab = AutoTokenizer.from_pretrained("facebook/mms-tts-ara", cache_dir=models_dir)
        self.model_arab.to(self.device).eval()

        self.model_deu = VitsModel.from_pretrained("facebook/mms-tts-deu", cache_dir=models_dir)
        self.tokenizer_deu = AutoTokenizer.from_pretrained("facebook/mms-tts-deu", cache_dir=models_dir)
        self.model_deu.to(self.device).eval()

        self.model_rus = VitsModel.from_pretrained("facebook/mms-tts-rus", cache_dir=models_dir)
        self.tokenizer_rus = AutoTokenizer.from_pretrained("facebook/mms-tts-rus", cache_dir=models_dir)
        self.model_rus.to(self.device).eval()

        self.model_hin = VitsModel.from_pretrained("facebook/mms-tts-hin",cache_dir=models_dir)
        self.tokenizer_hin = AutoTokenizer.from_pretrained("facebook/mms-tts-hin", cache_dir=models_dir)
        self.model_hin.to(self.device).eval()

        self.model_ben = VitsModel.from_pretrained("facebook/mms-tts-ben",cache_dir=models_dir)
        self.tokenizer_ben = AutoTokenizer.from_pretrained("facebook/mms-tts-ben", cache_dir=models_dir)
        self.model_ben.to(self.device).eval()

        self.model_mya = VitsModel.from_pretrained("facebook/mms-tts-mya",cache_dir=models_dir)
        self.tokenizer_mya = AutoTokenizer.from_pretrained("facebook/mms-tts-mya", cache_dir=models_dir)
        self.model_mya.to(self.device).eval()

        self.model_nld = VitsModel.from_pretrained("facebook/mms-tts-nld",cache_dir=models_dir)
        self.tokenizer_nld = AutoTokenizer.from_pretrained("facebook/mms-tts-nld", cache_dir=models_dir)
        self.model_nld.to(self.device).eval()

        self.model_fin = VitsModel.from_pretrained("facebook/mms-tts-fin",cache_dir=models_dir)
        self.tokenizer_fin = AutoTokenizer.from_pretrained("facebook/mms-tts-fin", cache_dir=models_dir)
        self.model_fin.to(self.device).eval()

        self.model_heb = VitsModel.from_pretrained("facebook/mms-tts-heb",cache_dir=models_dir)
        self.tokenizer_heb = AutoTokenizer.from_pretrained("facebook/mms-tts-heb", cache_dir=models_dir)
        self.model_heb.to(self.device).eval()

        self.model_ind = VitsModel.from_pretrained("facebook/mms-tts-ind",cache_dir=models_dir)
        self.tokenizer_ind = AutoTokenizer.from_pretrained("facebook/mms-tts-ind", cache_dir=models_dir)
        self.model_ind.to(self.device).eval()

        self.model_spa = VitsModel.from_pretrained("facebook/mms-tts-spa",cache_dir=models_dir)
        self.tokenizer_spa = AutoTokenizer.from_pretrained("facebook/mms-tts-spa", cache_dir=models_dir)
        self.model_spa.to(self.device).eval()

        self.model_kan = VitsModel.from_pretrained("facebook/mms-tts-kan",cache_dir=models_dir)
        self.tokenizer_kan = AutoTokenizer.from_pretrained("facebook/mms-tts-kan", cache_dir=models_dir)
        self.model_kan.to(self.device).eval()

        self.model_zlm = VitsModel.from_pretrained("facebook/mms-tts-zlm",cache_dir=models_dir)
        self.tokenizer_zlm = AutoTokenizer.from_pretrained("facebook/mms-tts-zlm", cache_dir=models_dir)
        self.model_zlm.to(self.device).eval()

        self.tts_eng("Preloading the model once, as you've shown")

    
    def voice_conv(self, out_array):
        data = (out_array, 16_000)
        result_array, sample_rate = self.converter.generate_from_cache(
            audio_data=data,
            tag=self.converter_tag,
        )
        return result_array, sample_rate


    def tts_urd(self, text):
        inputs = self.tokenizer_urd(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_urd(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()

    def tts_eng(self, text):
        inputs = self.tokenizer_eng(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_eng(**inputs.to(self.device))
        return self.voice_conv(outputs.waveform[0].cpu().float().numpy())

    def tts_ara(self, text):
        inputs = self.tokenizer_arab(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_arab(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()

    def tts_deu(self, text):
        inputs = self.tokenizer_deu(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_deu(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()

    def tts_rus(self, text):
        inputs = self.tokenizer_rus(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_rus(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()
    
    def tts_hin(self, text):
        inputs = self.tokenizer_hin(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_hin(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()
    
    def tts_ben(self, text):
        inputs = self.tokenizer_ben(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_ben(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()
    
    def tts_mya(self, text):
        inputs = self.tokenizer_mya(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_mya(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()
    
    def tts_nld(self, text):
        inputs = self.tokenizer_nld(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_nld(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()
    
    def tts_fin(self, text):
        inputs = self.tokenizer_fin(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_fin(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()
    
    def tts_heb(self, text):
        inputs = self.tokenizer_heb(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_heb(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()
    
    def tts_ind(self, text):
        inputs = self.tokenizer_ind(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_ind(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()
    
    def tts_spa(self, text):
        inputs = self.tokenizer_spa(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_spa(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()
    
    def tts_kan(self, text):
        inputs = self.tokenizer_kan(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_kan(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()
    
    def tts_zlm(self, text):
        inputs = self.tokenizer_zlm(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_zlm(**inputs.to(self.device))
        return outputs.waveform[0].cpu().float().numpy()