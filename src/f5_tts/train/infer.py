#%%
import sys
import os
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import sys
import os
# Add the parent directory of f5_tts to Python path
sys.path.append(os.path.abspath("/root/workdir/skrivtesnakk/F5-TTS/src/"))

from f5_tts.train.train_pl import LitCFMModel  # ensure correct 
from f5_tts.model import CFM, DiT, UNetT
from f5_tts.model.utils import get_tokenizer
from f5_tts.infer.utils_infer import load_vocoder  # Assuming this function is available
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
import re


def main_process(ref_audio, ref_text, text_gen, model_obj, mel_spec_type, remove_silence, speed):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if True:# "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        print("Voice:", voice)
        print("Ref_audio:", voices[voice]["ref_audio"])
        print("Ref_text:", voices[voice]["ref_text"])

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, text_gen)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        gen_text = text.strip()
        ref_audio = voices[voice]["ref_audio"]
        ref_text = voices[voice]["ref_text"]
        print(f"Voice: {voice}")
        audio, final_sample_rate, spectragram = infer_process(
            ref_audio, ref_text, gen_text, model_obj, vocoder, mel_spec_type=mel_spec_type, speed=speed
        )
        generated_audio_segments.append(audio)

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)
        return final_wave
    else:
        return None

#%% 
ref_audio = "/root/workdir/skrivtesnakk/tests/simen-bor-i-drammen.wav"
ref_text = "Hei, jeg heter Simen og jeg bor i Drammen."
gen_text = "Velkommen til VGs valgstudio. På tirsdag kommer det overraskende resultatet at Harris' og Trump går i samarbeid og danner presidentskap sammen."
remove_silence=False
vocoder_name = "vocos"
mel_spec_type = vocoder_name
speed = 1.0
output_dir = "/root/workdir/skrivtesnakk/tests"
wave_path = Path(output_dir) / f"output.wav"
#%% LOAD VOCODER
if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"

vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False, local_path=vocoder_local_path)

#%% LOAD MODEL
# Specify the path to the saved checkpoint
checkpoint_path = "/root/workdir/skrivtesnakk/F5-TTS/ckpts/F5-mini/epoch=0-step=10000.ckpt"

# Load the LitCFMModel from the checkpoint
pl_module = LitCFMModel.load_from_checkpoint(checkpoint_path)
model = pl_module.model
# Use the model as needed, e.g., for inference
#%%
final_wave = main_process(ref_audio, ref_text, gen_text, model, mel_spec_type, remove_silence, speed)
# %%
from IPython.display import Audio, display
# Assuming final_wave is your numpy array and final_sample_rate is the sample rate of the audio
audio = Audio(data=final_wave, rate=24000)

# Display the audio widget to play the sound in the notebook
display(audio)
# %%
