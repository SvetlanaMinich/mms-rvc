from TTS.api import TTS
import torch
import time

#  /root/.local/share/tts/voice_conversion_models--multilingual--vctk--freevc24
#  /root/.local/share/tts/wavlm/WavLM-Large.pt

device = "cuda:0" if torch.cuda.is_available() else "cpu"


start = time.time()
tts.voice_conversion_to_file(source_wav="Faib.wav", target_wav="output0.wav", file_path="faib-to-mal.wav")
print(f'     > Time: {time.time() - start}')