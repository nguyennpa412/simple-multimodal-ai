import os
import gdown
import shutil
from huggingface_hub import snapshot_download

class Resources():
    def __init__(self, speakers_type: str) -> None:
        self.home_dir = os.environ["APP_HOME_DIR"]
        self.speakers_type = speakers_type
        self.resources = ("xtts_v2", "speakers")
        self.resources_dir = {
            self.resources[0]: f"{self.home_dir}/src/modules/text_to_speech/xtts_v2/",
            self.resources[1]: f"{self.home_dir}/assets/audios/speakers/{self.speakers_type}/"
        }
        self.required_files = {
            self.resources[0]: {"model.pth", "config.json", "vocab.json"},
            self.resources[1]: {"david_attenborough.wav", "morgan_freeman.wav"},
        }
        self.xtts_v2_repo_id = "coqui/XTTS-v2"
        self.speakers_gdrive_id = {
            "short": "1l0kIf0Lm9B7O1BFpW-2G-q0jfE86CZ92",
            "long": "1732Cqn21Azu-QN3HbNZgPojShV-cT0Hs"    
        }
        self.speakers_gdown_output = f"{self.home_dir}/assets/audios/"

    def check_required_files(self) -> None:
        for resource in self.resources:
            print(f"\n> CHECK RESOURCES: {resource}...")
            try:
                resource_files = set(os.listdir(self.resources_dir[resource]))
            except:
                resource_files = set()
            missing_files = self.required_files[resource] - resource_files
            if len(missing_files) > 0:
                self.get(resource=resource)
            print(f"> DONE CHECKING RESOURCES: {resource} !")

    def get(self, resource: str) -> None:
        if resource == self.resources[0]:
            print(">> GET XTTS-V2 resources...")
            snapshot_download(
                repo_id=self.xtts_v2_repo_id,
                local_dir=self.resources_dir[resource],
                allow_patterns=["*.json", "model.pth"]
            )

        if resource == self.resources[1]:
            print(">> GET SPEAKERS wavs...")
            speakers_file_path = gdown.download(
                id=self.speakers_gdrive_id[self.speakers_type],
                output=self.speakers_gdown_output
            )
            shutil.unpack_archive(
                filename=speakers_file_path,
                extract_dir=self.speakers_gdown_output
            )
            os.remove(path=speakers_file_path)

            if self.speakers_type == "short":
                shutil.rmtree(path=f"{self.home_dir}/assets/audios/speakers/long/", ignore_errors=True)
            else:
                shutil.rmtree(path=f"{self.home_dir}/assets/audios/speakers/short/", ignore_errors=True)