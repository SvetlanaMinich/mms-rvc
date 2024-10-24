from styletts2 import tts
import time
import nltk
nltk.download('punkt_tab')


other_tts = tts.StyleTTS2(model_checkpoint_path='libritts/epochs_2nd_00020.pth', config_path='libritts/config.yml')

en_test = ['A small man in a holey yellow bowler hat and a pear-shaped crimson nose.',
    'In checkered trousers and patent leather boots rode onto the stage on an ordinary two-wheeled bicycle.',
    'He made a circle to the sound of a foxtrot.', 
    'And then let out a triumphant cry, causing the bicycle to rear up.',
    'Having ridden on one rear wheel, the man turned upside down.', 
    'Managed to unscrew the front wheel while moving and let it go behind the scenes.',
    'And then continued on one wheel, turning the pedals with his hands.',
    "On a tall metal pole with a saddle on top,",
    "a full blonde in tights rode out with one wheel.",
    "Her skirt was studded with silver stars.",
    "She started riding in circles around the arena.",
    "The man greeted her with loud shouts.",
    "With his foot, he knocked off his hat when meeting.",
    "Finally, a little boy around eight years old rode out.",
    "He had an old man’s face and a tiny bike.",
    "The bike had a big car horn attached to it.",
    "The boy weaved between adults on the stage.",
    "Under the drumroll, they approached the stage edge.",
    "The audience gasped and leaned back in fear.",
    "They thought all three would fall into the orchestra.",
    ]
for i, sen in enumerate(en_test):
    start = time.time()
    other_tts.inference(sen, target_voice_path="def_male.wav", output_wav_file=f"eng-{i}.wav")
    print(f'        > time: {time.time() - start}')
