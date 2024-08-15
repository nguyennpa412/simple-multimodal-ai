import os
import time
import requests
import numpy as np
import gradio as gr

from PIL import Image
from io import BytesIO
from src.utils.loader import AppLoader
from src.utils.base_utils import BaseUtils

class VisualInputUtils(BaseUtils):
    def __init__(self, app_loader: AppLoader) -> None:
        self.app_loader = app_loader
        self.home_dir = os.environ["APP_HOME_DIR"]
        self.curr_tasks = self.app_loader.curr_tasks
        self.num_task = self.app_loader.num_task

    def select_demo_image(self,
        image: np.ndarray,
        demo_images: list,
        select_event: gr.SelectData
    ) -> tuple[dict, dict, dict]:
        img_idx = select_event.index
        if self.app_loader.visinp_demo_images_interactive | (image is None):
            self.app_loader.visinp_selected_demo_image_idx = img_idx
            img_path = demo_images[img_idx][0]
            return gr.update(value=img_path), gr.update(), None
        else:
            gr.Warning(message="Please clear the Input Image first!")
            return gr.update(), gr.update(selected_index=None), gr.update()

    def get_bot_avatar(self, timeout: float = 2) -> None:
        try:
            response = requests.get(
                url="https://robohash.org/{timestamp}".format(timestamp=time.time()),
                timeout=timeout
            )
            self.app_loader.bot_avatar = Image.open(BytesIO(response.content))
        except:
            self.app_loader.bot_avatar = Image.open(self.app_loader.default_bot_avatar_path)

        self.app_loader.bot_avatar.save(
            fp=self.app_loader.tmp_bot_avatar_path,
            format="png"
        )
        self.done_task(task="get_botavatar")

    def get_description(self, image: np.ndarray = None) -> str:
        description = None
        if image is not None:
            self.start_task(task="get_description")
            self.start_task(task="get_botavatar")
            try:
                description = self.app_loader.modules["ittt"].run_model(
                    visual_input=image,
                    text_input="!describe -m"
                )["answer"]
            except:
                description = ""

            if description is not None:
                if description.strip() == "":
                    description = "Description is not available at the moment!"
                while "get_botavatar" in self.curr_tasks:
                    time.sleep(0.1)

        return description

    def get_speech(self, text: str = None, audio: tuple = None) -> tuple[int, np.ndarray]:
        speech = None
        if text is not None:
            if text.strip() != "":
                if audio is None:
                    self.start_task(task="get_speech")
                    try:
                        speech = self.app_loader.modules["tts"].run_model(text=text)
                    except:
                        speech = f"{self.home_dir}/assets/audios/not_available.wav"

        return speech

    def change_speech_icon(self, text: str = None, audio: tuple = None) -> tuple[dict, dict]:
        icon_path = f"{self.home_dir}/assets/icons/audio_play.png"
        bot_avatar = Image.new('RGBA', (300, 300), (255, 0, 0, 0))
        interactive = False
        if text is not None:
            if text.strip() != "":
                if audio is not None:
                    icon_path = f"{self.home_dir}/assets/icons/audio_stop.png"
                    bot_avatar = self.app_loader.bot_avatar
                else:
                    self.done_task(task="play_speech")
                    
                    if self.num_task == 0:
                        interactive=True

        return gr.update(icon=icon_path), \
                gr.update(value=bot_avatar), \
                gr.update(interactive=interactive)

    def input_image_handle(self, image: np.ndarray) -> dict:
        update_annotated_image = gr.update()
        update_annotated_image_tab = gr.update()
        update_image_tabs = gr.update()
        
        if image is None:
            self.reset_task_tracker()
            try:
                self.app_loader.modules["ittt"].clear_vision_tasks_history()
            except:
                pass
            self.app_loader.visinp_demo_images_interactive = True
            update_annotated_image = gr.update(value=None, visible=False)
            update_annotated_image_tab = gr.update(visible=False)
            update_image_tabs = gr.update(selected=0)
        else:
            self.get_bot_avatar()
            self.app_loader.visinp_demo_images_interactive = False

        return  gr.update(interactive=False),\
                update_annotated_image,\
                update_annotated_image_tab,\
                update_image_tabs

    def synth_handle(self,
        text: str = None,
        audio: tuple = None,
        speech_btn: any = None,
        clear_image_btn: any = None
    ) -> tuple[dict, dict]:
        if text is not None:
            if text.strip() != "":
                if audio is None:
                    icon_path = f"{self.home_dir}/assets/icons/audio_synth.gif"

                    return gr.update(icon=icon_path), gr.update(interactive=False)

        return speech_btn, clear_image_btn

    def description_handle(self,
        text: str = None,
        audio: tuple = None
    ) -> tuple[dict, dict]:
        speech = None
        interactive = False
        if text is not None:
            if text.strip() != "":
                self.done_task(task="get_description")
                speech = audio
                
                if self.num_task == 0:
                    interactive=True

        return speech, gr.update(interactive=interactive)

    def release_interactive(self) -> None:
        interactive = False
        self.done_task(task="get_speech")
        self.start_task(task="play_speech")

        return gr.update(interactive=interactive)

    def select_speaker(self, select_event: gr.SelectData) -> None:
        try:
            self.app_loader.modules["tts"].set_speaker(name_or_index=select_event.index)
        except:
            pass
