import os
import time
import torch
import base64
import random
import numpy as np
import gradio as gr

from PIL import Image
from io import BytesIO
from src.utils.loader import AppLoader
from src.utils.base_utils import BaseUtils
from src.components.visual_input import VisualInput

GREETING_MSG = [
    "Hi!",
    "Hello!",
    "Greetings!",
    "🤖👋",
    "👋🤖",
    "🤖💡",
    "🤖💬",
    "👀💭",
    "[┐∵]┘",
    "♪[┐∵]┘♪",
    "└[∵┌]",
    "♪└[∵┌]♪",
    "└[ ∵ ]┘",
    "♪┌| ∵ |┘♪",
    ""
]

FIRST_MSG = [
    "It is a nice day for ImageQA",
    "How can I help you?",
    "How can I help you with this image?",
    "What do you want to know about this image?",
    ""
]

class ChatBotUtils(BaseUtils):
    def __init__(self,
        app_loader: AppLoader,
        visual_input: VisualInput
    ) -> None:
        self.app_loader = app_loader
        self.visual_input = visual_input
        self.home_dir = os.environ["APP_HOME_DIR"]
        self.current_chat_length = 0
        self.is_update_annotated_image = False
        self.curr_tasks = self.app_loader.curr_tasks
        self.num_task = self.app_loader.num_task
    
    def update_chatbot_first_interaction(self,
        image: np.ndarray,
        description: str
    ) -> tuple[dict, dict]:
        input_interactive = False
        if (image is not None) & (description is not None):
            if description.strip() != "":
                bot_message = ""
                while bot_message.strip() == "":
                    random.seed(time.time())
                    greeting = random.choice(seq=GREETING_MSG)
                    first_message = random.choice(seq=FIRST_MSG)
                    bot_message = f"{greeting}\n{first_message}"
                    
                input_interactive = True
                return [gr.update(
                    value=[(None, bot_message)],
                    avatar_images=(None, self.app_loader.tmp_bot_avatar_path)
                ), gr.update(
                    interactive=input_interactive,
                    autofocus=True,
                    placeholder="Enter message...",
                    submit_btn=True
                )] + [gr.update(interactive=input_interactive)]*self.quick_command_component_num
            
        self.current_chat_length = 0

        try:
            self.app_loader.modules["vqa"].clear_chat(chat_mode=self.app_loader.modules["vqa"].chat_mode)
        except:
            pass
        
        return [gr.update(value=None), gr.update(
            value=None,
            interactive=input_interactive,
            autofocus=False,
            placeholder="",
            submit_btn=False,
        )] + [gr.update(interactive=input_interactive)]*self.quick_command_component_num
        
    def add_message(self,
        image: np.ndarray,
        description: str,
        message: dict,
        chat_history: list
    ) -> tuple:
        self.is_update_annotated_image = False
        textbox_autofocus = True
        update_annotated_image = gr.update()
        update_annotated_image_tab = gr.update()
        update_image_tabs = gr.update()
        update_annot_status = gr.update()
        update_clear_input_image_btn = gr.update()
        
        # for x in message["files"]:
        #     chat_history.append([(x,), None])

        warning_message = "Missing input text!"
        text_input = message["text"]

        if image is not None:
            if description is not None:
                if description.strip() != "":
                    if text_input is not None:
                        if type(text_input) is str:
                            if text_input.strip() != "":
                                textbox_autofocus = False
                                bot_msg = None
                                warning_message = None
                                update_clear_input_image_btn = gr.update(interactive=False)
                                self.start_task(task="chatbot")
                                
                                try:
                                    self.task_extraction = self.app_loader.modules["ittt"].extract_task(text_input=text_input)
                                except:
                                    self.task_extraction = None
                                
                                if self.task_extraction is not None:
                                    if (
                                        self.task_extraction["has_command_sign"]
                                        & (self.task_extraction["task"] is not None)
                                        & (self.task_extraction["prompt"] is not None)
                                        & (self.task_extraction["command"] is not None)
                                    ):
                                        command = self.task_extraction["command"]
                                        if command in self.app_loader.modules["ittt"].is_annot_commands:
                                            update_annotated_image = gr.update(
                                                value=None,
                                                color_map=None,
                                                visible=False
                                            )
                                            update_annotated_image_tab = gr.update(visible=False)
                                            update_image_tabs = gr.update(selected=0)
                                            update_annot_status = gr.update(
                                                value=__class__.get_annot_status_value(
                                                    url=self.annot_status_icon_url,
                                                    text=f"Processing {command} task..."
                                                ),
                                                visible=True
                                            )
                                            if command == "segment":
                                                bot_msg = "This task usually takes long time to process, please be patient !"
                                
                                chat_history.append((text_input, bot_msg))

                else:
                    warning_message = "Wait for description!"
            else:
                warning_message = "Wait for description!"
        else:
            warning_message = "Missing input image!"

        if warning_message is not None:
            gr.Warning(message=warning_message)

        return [
            gr.update(value=None, autofocus=textbox_autofocus), \
            chat_history, \
            update_annotated_image, \
            update_annotated_image_tab, \
            update_image_tabs, \
            update_annot_status, \
            update_clear_input_image_btn
        ]  + [gr.update(interactive=False)]*self.quick_command_component_num
    
    def respond(self,
        image: np.ndarray,
        chat_history: list
    ) -> tuple:
        chat_length = len(chat_history)
        if (chat_length > 0) & (chat_length > self.current_chat_length):
            message = chat_history[-1][0]
            if (image is not None) & (message is not None):
                if message.strip() != "":
                    try:
                        vision_task_answer = None
                        try:
                            if self.task_extraction["has_command_sign"]:
                                vision_task_answer = self.app_loader.modules["ittt"].run_model(
                                    visual_input=image,
                                    text_input=message,
                                    task_extraction=self.task_extraction
                                )
                        except:
                            pass

                        if vision_task_answer is not None:
                            bot_message = vision_task_answer["answer"]
                            self.is_update_annotated_image = vision_task_answer["is_update_annotated_image"]
                            self.update_annotated_value = vision_task_answer["update_annotated_value"]
                            self.update_annotated_color_map = vision_task_answer["update_annotated_color_map"]
                        else:
                            bot_message = self.app_loader.modules["vqa"].run_model(
                                visual_input=image,
                                question=message
                            )
                    except torch.cuda.OutOfMemoryError:
                        bot_message = "CUDA is currently out of memory!\nPlease begin a new chat to continue chatting, sorry for the inconvenience."
                        time.sleep(1.5)
                    except:
                        bot_message = "Please try again !"
                        time.sleep(1.5)

                    chat_history[-1][1] = bot_message
                    self.current_chat_length = len(chat_history)

        return [None, chat_history] + [gr.update(interactive=True)]*self.quick_command_component_num

    def update_annot_status(self) -> dict:
        update_annot_status = gr.update(
            value=None,
            visible=False
        )

        if self.is_update_annotated_image:
            update_annot_status = gr.update(
                value=__class__.get_annot_status_value(
                    url=self.annot_status_icon_url,
                    text=f"Generating Annotated Image..."
                ),
                visible=True
            )

        return update_annot_status
    
    def update_annotated(self) -> tuple:
        update_annotated_image = gr.update()
        update_annotated_image_tab = gr.update()
        update_image_tabs = gr.update()

        if self.is_update_annotated_image:
            update_annotated_image = gr.update(
                value=self.update_annotated_value,
                color_map=self.update_annotated_color_map,
                visible=True
            )
            update_annotated_image_tab = gr.update(visible=True)
            update_image_tabs = gr.update(selected=1)
            self.is_update_annotated_image = False

        return update_annotated_image,\
            update_annotated_image_tab,\
            update_image_tabs
            
    def postprocess(self) -> tuple:
        interactive = False
        self.done_task(task="chatbot")
        
        if self.num_task == 0:
            interactive=True
        return gr.update(value=None, visible=False), gr.update(interactive=interactive)

    @staticmethod
    def img_2_b64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode(encoding="utf-8")

        return img_str

    @staticmethod
    def get_annot_status_value(url: str, text: str, width: int = 32) -> str:
        return f"""
            <div id="chatbot-annot-status-md-div">
                <img src="{url}" alt="Loading..." width={width} />
                <span>{text}</span>
            </div>
        """