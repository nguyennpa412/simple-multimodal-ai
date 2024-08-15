import os
import yaml

import pandas as pd
from PIL import Image
from src.modules.text_to_speech.xtts_v2 import SpeechSynthesizer
from src.modules.visual_question_answering.mini_internvl_chat_2b_v1_5 import VQAChatbot
from src.modules.image_text_to_text.florence_2_large import ComputerVisionTasks

TASK_MAP = {
    "tts": SpeechSynthesizer,
    "vqa": VQAChatbot,
    "ittt": ComputerVisionTasks,
}
SPECS_TO_SPEAKERS = {
    "cpu": "short",
    "gpu_low": "short",
    "gpu_high": "long"
}

class AppLoader():
    def __init__(self, app_config_path: str) -> None:
        self.app_config_path = app_config_path
        self.home_dir = os.environ["APP_HOME_DIR"]
        self.curr_tasks = set()
        self.num_task = 0
        self.modules = {}

    def set_attributes(self, **kwargs) -> None:
        for key in kwargs:
            self.__setattr__(key, kwargs[key])

    def load_config(self) -> None:
        print(f"> LOAD APP CONFIG...")
        config = __class__.load_yaml(yaml_path=self.app_config_path)
        try:
            self.module_config = config["module_config"]
        except:
            self.module_config = []
        self.set_attributes(**config["app_config"])
        self.speakers_type = SPECS_TO_SPEAKERS[self.specs_type]
        self.css_path = f"{self.home_dir}/{self.css_path}"
        self.launch_config = config["launch_config"]
        self.launch_config["favicon_path"] = self.home_dir + "/" + self.launch_config["favicon_path"]

        self.demo_image_path = f"{self.home_dir}/{self.demo_image_path}"
        self.demo_images_folder_path = f"{self.home_dir}/{self.demo_images_folder_path}"
        self.demo_images_path_list = [
            (f"{self.demo_images_folder_path}/{img_file}", None) for img_file in
            sorted(os.listdir(self.demo_images_folder_path))
        ]

        self.default_bot_avatar_path = f"{self.home_dir}/{self.default_bot_avatar_path}"
        self.bot_avatar = Image.new('RGBA', (300, 300), (255, 0, 0, 0))
        if not os.path.exists(os.path.dirname(self.default_bot_avatar_path) + "/tmp/"):
            os.mkdir(os.path.dirname(self.default_bot_avatar_path) + "/tmp/")
        self.tmp_bot_avatar_path = os.path.dirname(self.default_bot_avatar_path) + "/tmp/tmp_bot.png"
        Image.open(fp=self.default_bot_avatar_path).save(fp=self.tmp_bot_avatar_path, format="png")

        self.vision_tasks_config_path = f"{self.home_dir}/{self.vision_tasks_config_path}"
        os.environ["VISION_TASKS_CONFIG_PATH"] = self.vision_tasks_config_path
        self.vision_tasks_config = pd.read_parquet(self.vision_tasks_config_path)
        self.vision_tasks_config = self.vision_tasks_config[self.vision_tasks_config["usage"] != "!detect -a"].copy()

    def load_modules(self, ignore_module: str = None) -> None:
        print(f"\n> LOAD MODULES...")
        for task in self.module_config:
            if self.specs_type == "cpu":
                if task == "vqa":
                    continue
            if (ignore_module is None) | (task != ignore_module):
                config = __class__.load_yaml(
                    yaml_path="{home_dir}/{config_path}".format(
                        home_dir=self.home_dir,
                        config_path=self.module_config[task]
                    )
                )
                config["load_config"]["specs_type"] = self.specs_type
                if task == "tts":
                    config["load_config"]["speakers_type"] = self.speakers_type
                
                self.modules[task] = TASK_MAP[task](config=config)

    @staticmethod
    def load_yaml(yaml_path: str) -> any:
        with open(yaml_path, "r") as yamlfile:
            return(yaml.load(stream=yamlfile, Loader=yaml.FullLoader))