import os
import time
import torch
import numpy as np

from typing import Union, Literal
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.utils.audio.numpy_transforms import save_wav

from src.modules.base_module import BaseModule

class SpeechSynthesizer(BaseModule):
    def __init__(self, config: dict) -> None:
        print("\n" + "==||"*20 + "==")
        print("> INIT SpeechSynthesizer...")
        self.set_attributes(**config["load_config"])
        self.set_attributes(**config["load_config"]["specs"][self.specs_type])
        self.speakers_dir_path = os.environ["APP_HOME_DIR"] + f"/{self.speakers_dir_path}/{self.speakers_type}"
        self.model_dir = os.environ["APP_HOME_DIR"] + "/" + self.model_dir_path
        self.warm_up_text = config["warm_up_config"]["text"]
        self.warm_up_output_wav_path = os.environ["APP_HOME_DIR"] + "/" + config["warm_up_config"]["output_wav_path"]

        self.clear_synth_data()
        self.load_speakers()
        self.set_speaker(name_or_index=0)
        self.set_speaker_wav_path()
        self.load_device()
        self.start_module()

    def load_speakers(self) -> None:
        print("> LOAD SPEAKERS...")
        self.speakers = tuple(sorted([f.split(".wav")[0] for f in os.listdir(self.speakers_dir_path) if ".wav" in f]))
        speakers_idx = tuple(range(len(self.speakers)))
        self.speaker_to_idx = dict(zip(
            self.speakers,
            speakers_idx,
        ))
        self.idx_to_speaker = dict(zip(
            speakers_idx,
            self.speakers,
        ))

    def get_speakers(self) -> list[str]:
        return self.speakers

    def set_speaker(self,
        name_or_index: Union[str, int] = 0,
        by: Literal["name", "index"] = "index"
    ) -> None:
        try:
            if by == "index":
                speaker = self.idx_to_speaker[int(name_or_index)]
            elif by == "name":
                if str(name_or_index) in self.speakers:
                    speaker = str(name_or_index)
        except:
            speaker = None
        
        if speaker is not None:
            self.speaker = speaker
            print(f"> SET SPEAKER -> {self.speaker}")
            self.set_speaker_wav_path()

    def set_speaker_wav_path(self) -> None:
        self.speaker_wav_path = f"{self.speakers_dir_path}/{self.speaker}.wav"

    def load_config(self) -> None:
        print("> LOAD CONFIG...")
        self.config = XttsConfig()
        self.config.load_json(file_name=f"{self.model_dir}/config.json")

    def load_model(self) -> None:
        print("> LOAD MODEL...")
        self.set_seed(seed=42)
        self.model = Xtts.init_from_config(config=self.config)
        self.model.load_checkpoint(config=self.config, checkpoint_dir=self.model_dir, eval=True)
        self.model = self.model.to(device=self.device)

    def insert_synth_data(self,
        key: str,
        speaker: str,
        text: str,
        wav_data: tuple[int, np.ndarray]
    ) -> None:
        self.synth_data[key] = {
            "speaker": speaker,
            "text": text,
            "wav_data": wav_data
        }

    def clear_synth_data(self) -> None:
        self.synth_data = {}

    def start_module(self, warm_up: bool = True) -> None:
        print("# START MODULE...")
        self.load_config()
        self.load_model()
        if warm_up:
            print("> WARM UP...")
            self.run_model(
                text=self.warm_up_text,
                output_wav_path=self.warm_up_output_wav_path,
                save_wav_file=True,
                warm_up=True
            )

    def stop_module(self) -> None:
        print("# STOP MODULE...")
        self.config = None
        self.model = None
        self.clear_cache()

    def run_model(self,
        text: str,
        output_wav_path: str = None,
        save_wav_file: bool = False,
        warm_up: bool = False
    ) -> tuple[int, np.ndarray]:
        if text is not None:
            if text.strip() != "":
                key_str = f"[{self.speaker}] {text}"
                hashed_key = __class__.hash_text(key_str)

                if hashed_key in self.synth_data:
                    wav_data = self.synth_data[hashed_key]["wav_data"]
                else:
                    print("\n" + "-"*50)
                    print(f"> SpeechSynthesizer: {text}")
                    start_time = time.time()

                    with torch.inference_mode():
                        outputs = self.model.synthesize(
                            text=text,
                            config=self.config,
                            speaker_wav=self.speaker_wav_path,
                            gpt_cond_len=self.gpt_cond_len,
                            language=self.language,
                        )
                    self.clear_cache()

                    wav_data = (
                        self.config.get("audio")["output_sample_rate"],
                        outputs["wav"]
                    )
                    
                    end_time = time.time()
                    print(f"  >> Time elapsed: {end_time - start_time:.2f} s")
                    print("-"*50)

                    self.insert_synth_data(
                        key=hashed_key,
                        speaker=self.speaker,
                        text=text,
                        wav_data=wav_data
                    )

                if (
                    save_wav_file\
                    & (output_wav_path is not None)
                ):
                    if output_wav_path.strip() != "":
                        self.save_wav_file(wav_data=wav_data, output_wav_path=output_wav_path)

                if warm_up:
                    self.clear_synth_data()

                return wav_data

    def save_wav_file(self,
        wav_data: tuple[int, np.ndarray],
        output_wav_path: str
    ) -> None:
        save_wav(
            wav=wav_data[1],
            path=output_wav_path,
            sample_rate=wav_data[0]
        )
        print(f"\n> WAV FILE SAVED: {output_wav_path}")
