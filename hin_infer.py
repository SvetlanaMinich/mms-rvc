import os
from extra import TTSTokenizer, VitsConfig, CharactersConfig, VitsCharacters
from transformers import VitsModel
import torch
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("chars.txt", 'r') as f:
    letters = f.read().strip('\n')

model="hi_male_vits_30hrs.pt"
hin_test = [
    "छोटा आदमी फटे हुए पीले टोपी और नाशपाती जैसी लाल नाक के साथ।",
    "चेक पैंट और पॉलिश किए हुए जूते पहने हुए, वह मंच पर साधारण दो पहियों वाली साइकिल पर आया।",
    "फॉक्सट्रॉट संगीत के साथ उसने एक चक्कर लगाया।",
    "फिर उसने विजय की चिल्लाहट लगाई, जिससे साइकिल उठी।",
    "पीछे के पहिए पर सवारी करते हुए, आदमी उल्टा हो गया।",
    "उसने चलते-चलते आगे का पहिया हटा दिया और उसे पर्दे के पीछे भेजा।",
    "फिर उसने एक पहिए पर यात्रा जारी रखी, और हाथों से पैडल मारे।",
    "एक ऊँचे धातु के खंभे पर ऊपरी हिस्से में काठी थी।",
    "एक पहिए पर, एक पूरी गोरी महिला तंग कपड़े में आई।",
    "उसका स्कर्ट चांदी के सितारों से सजा हुआ था।",
    "वह सर्कल में घूमने लगी।",
    "आदमी ने जोर से चिल्ला कर उसे बधाई दी।",
    "मुलाकात के दौरान उसने पैर से टोपी हटा दी।",
    "अंत में, लगभग आठ साल का एक लड़का आ गया।",
    "उसका चेहरा बूढ़ा जैसा था और उसकी एक छोटी साइकिल थी।",
    "साइकिल में एक बड़ा कार हॉर्न लगा हुआ था।",
    "लड़का मंच पर बड़ों के बीच से घूम रहा था।",
    "ड्रम की धड़कन के साथ, वे मंच के किनारे पहुंचे।",
    "दर्शक डर से पीछे हट गए।",
    "उन्हें लगा कि तीनों ऑर्केस्ट्रा में गिर जाएंगे।",
]

config = VitsConfig(
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class=VitsCharacters,
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=letters,
        punctuations="!¡'(),-.:;¿? ",
        phonemes=None)
    )

tokenizer, config = TTSTokenizer.init_from_config(config)
net = VitsModel.from_pretrained(model).to(device).eval()

for i, sen in enumerate(hin_test):
    start = time.time()
    x = tokenizer.text_to_ids(sen)
    x = torch.from_numpy(np.array(x)).unsqueeze(0).to(device)
    with torch.no_grad():
        out2 = net(x)
    print(f'        > time: {time.time() - start}')
    # import soundfile as sf
    # sf.write(f"hin-{i}.wav", out2.squeeze().cpu().numpy(), 22050)