import os
import re
import copy
import time
import torch
import random
import numpy as np
import pandas as pd

from typing import Union
from hashlib import sha256
from unidecode import unidecode
from unittest.mock import patch
from PIL import Image, ImageDraw
from src.modules.base_module import BaseModule
from transformers.dynamic_module_utils import get_imports
from transformers.utils import is_flash_attn_2_available
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32
}

class ComputerVisionTasks(BaseModule):
    def __init__(self, config: dict) -> None:
        print("\n" + "==||"*20 + "==")
        print("> INIT ComputerVisionTasks...")
        self.set_attributes(**config["load_config"])
        self.set_attributes(**config["load_config"]["specs"][self.specs_type])
        self.warm_up_text_input = config["warm_up_config"]["text_input"]
        self.warm_up_image_path = os.environ["APP_HOME_DIR"] + "/" + config["warm_up_config"]["image_path"]
        self.model_dtype = MODEL_DTYPE[self.model_dtype]

        self.vision_tasks_history = {}
        self.vision_tasks_config_path = os.environ["VISION_TASKS_CONFIG_PATH"]
        self.vision_tasks_config = pd.read_parquet(self.vision_tasks_config_path)
        self.vision_tasks_commands = "|".join(set(self.vision_tasks_config["command"]))
        self.vision_tasks_task2command = dict(zip(
            self.vision_tasks_config["task"],
            self.vision_tasks_config["command"]
        ))
        self.vision_tasks_from_text_input_command = (
            "detect",
            "segment",
        )
        self.vision_tasks_from_text_input_task = (
            "<OPEN_VOCABULARY_DETECTION>",
            "<REGION_TO_SEGMENTATION>",
        )
        self.vision_tasks_from_text_input_command2task = dict(zip(
            self.vision_tasks_from_text_input_command,
            self.vision_tasks_from_text_input_task,
        ))
        self.vision_tasks_usage2task = dict(zip(
            self.vision_tasks_config["usage"],
            self.vision_tasks_config["task"]
        ))
        self.is_annot_commands = set(self.vision_tasks_config[self.vision_tasks_config["is_annot"]]["command"])

        self.load_device()
        self.start_module()

    def load_config(self) -> None:
        print("> LOAD CONFIG...")
        if self.load_in_4bit | self.load_in_8bit:
            if self.load_in_4bit & self.load_in_8bit:
                print("> USING load_in_8bit")
                self.load_in_4bit = False
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit
            )

    def load_model(self) -> None:
        print("> LOAD MODEL...")
        self.set_seed(seed=42)

        if self.load_in_4bit | self.load_in_8bit:
            assert self.device != "cpu", "Can't run quantized model on CPU !!!"
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_hf_path,
                quantization_config=self.quantization_config,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                trust_remote_code=self.trust_remote_code,
                device_map=self.device,
                torch_dtype=self.model_dtype
            ).eval()
        else:
            if is_flash_attn_2_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.model_hf_path,
                    low_cpu_mem_usage=self.low_cpu_mem_usage,
                    trust_remote_code=self.trust_remote_code,
                    device_map=self.device,
                    torch_dtype=self.model_dtype
                ).eval()
            else:
                #workaround for unnecessary flash_attn requirement
                with patch("transformers.dynamic_module_utils.get_imports", __class__.fixed_get_imports):
                    self.model = AutoModelForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=self.model_hf_path,
                        attn_implementation="sdpa",
                        low_cpu_mem_usage=self.low_cpu_mem_usage,
                        trust_remote_code=self.trust_remote_code,
                        device_map=self.device,
                        torch_dtype=self.model_dtype,
                    ).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=self.model_hf_path, trust_remote_code=self.trust_remote_code)

    def start_module(self) -> None:
        print("# START MODULE...")
        self.load_config()
        self.load_model()
        print("> WARM UP...")
        print(
            "\n> WARM UP OUTPUT:",
            self.run_model(
                visual_input=self.warm_up_image_path,
                text_input=self.warm_up_text_input,
            )
        )

    def stop_module(self) -> None:
        print("# STOP MODULE...")
        self.model = None
        self.processor = None
        self.clear_cache()

    def clear_vision_tasks_history(self) -> None:
        self.vision_tasks_history = {}

    def run_model(self,
        visual_input: Union[str, np.ndarray],
        text_input: str,
        task_extraction: dict = None
    ) -> any:
        vision_task_answer = None
        if (
            (visual_input is not None)\
            & (text_input is not None)
        ):
            if task_extraction is None:
                task_extraction = self.extract_task(text_input=text_input)

            if task_extraction["has_command_sign"]:
                vision_task_answer = {
                    "command": None,
                    "answer": "Invalid command !",
                    "is_update_annotated_image": False,
                    "update_annotated_value": None,
                    "update_annotated_color_map": None,
                }
                task = task_extraction["task"]
                prompt = task_extraction["prompt"]
                
                if task is not None:
                    if prompt is not None:
                        command = task_extraction["command"]
                        vision_task_answer["command"] = command
                        image = __class__.get_pil_image(image=visual_input)
                        hashed_image = sha256(str(np.array(image).flatten().tolist()).encode("utf8")).hexdigest()
                        if hashed_image not in self.vision_tasks_history:
                            self.vision_tasks_history[hashed_image] = {}
                        
                        if command not in ("detect", "segment"):
                            print("\n" + "-"*50)
                            print(f"> ComputerVisionTasks: {prompt}")
                            start_time = time.time()
                            
                            task_res = self.get_history_data(
                                image=image,
                                hashed_image=hashed_image,
                                task=task,
                                prompt=prompt,
                                command=command
                            )

                            end_time = time.time()
                            print(f"  >> Time elapsed: {end_time - start_time:.2f} s")
                            print("-"*50)

                            if command == "describe":
                                vision_task_answer["answer"] = task_res
                                
                            elif command in ("densecap", "ocr"):
                                vision_task_answer = __class__.process_bbox(
                                    image=image,
                                    text_input=text_input,
                                    task_res=task_res,
                                    task_extraction=task_extraction,
                                    vision_task_answer=vision_task_answer,
                                )
                                
                            else:
                                vision_task_answer["answer"] = prompt

                        else:
                            if len(task_extraction["obj_list"]) > 0:
                                print("\n" + "-"*50)
                                start_time = time.time()
                                annots = []
                                image_size = [image.width, image.height]*2
                                
                                for obj in task_extraction["obj_list"]:
                                    prompt_=f"<OPEN_VOCABULARY_DETECTION>{obj}"
                                    task_="<OPEN_VOCABULARY_DETECTION>"
                                    command_ = "detect"
                                    if task_ not in self.vision_tasks_history[hashed_image]:
                                        self.vision_tasks_history[hashed_image][task_] = {}
                                        
                                    print(f"> ComputerVisionTasks: {prompt_}")
                                    
                                    task_res = self.get_history_data(
                                        image=image,
                                        hashed_image=hashed_image,
                                        task=task_,
                                        prompt=prompt_,
                                        command=command_,
                                        obj=obj
                                    )
                                    
                                    bboxes = task_res["bboxes"]
                                    labels = task_res["labels"]
                                    
                                    for idx, label in enumerate(labels):
                                        bbox = bboxes[idx]

                                        if command == "detect":
                                            annots.append((map(round, bbox), label))
                                        else:
                                            seg_task = "REGION_TO_SEGMENTATION"
                                            bbox_overlap_area, _ = __class__.calculate_overlap(
                                                bbox=bbox,
                                                image_size=[image.width, image.height]
                                            )
                                            if (bbox_overlap_area >= 0.55) & (len(labels) == 1):
                                                seg_task = "REFERRING_EXPRESSION_SEGMENTATION"

                                            if seg_task == "REGION_TO_SEGMENTATION":
                                                quantized_bbox = tuple(map(round, [bbox[i]*999/image_size[i] for i in range(4)]))
                                                loc = "".join(["<loc_{}>".format(x) for x in quantized_bbox])
                                            else:
                                                loc = obj
                                            
                                            prompt_=f"<{seg_task}>{loc}"
                                            task_=f"<{seg_task}>"
                                            command_ = "segment"
                                            if task_ not in self.vision_tasks_history[hashed_image]:
                                                self.vision_tasks_history[hashed_image][task_] = {}
                                                
                                            task_res = self.get_history_data(
                                                image=image,
                                                hashed_image=hashed_image,
                                                task=task_,
                                                prompt=prompt_,
                                                command=command_,
                                                obj=loc
                                            )
                                            
                                            mask = __class__.process_mask(image=image, polygons=task_res["polygons"])
                                            annots.append((mask, label))
                                    
                                end_time = time.time()
                                print(f"  >> Time elapsed: {end_time - start_time:.2f} s")
                                print("-"*50)
                                
                                vision_task_answer = __class__.postprocess(
                                    image=image,
                                    annots=annots,
                                    task_extraction=task_extraction,
                                    vision_task_answer=vision_task_answer
                                )
                            else:
                                vision_task_answer["answer"] = f"No result for this task: `{text_input[1:]}`"
                    else:
                        vision_task_answer["answer"] = f"No result for this task: `{text_input[1:]}`"

        return vision_task_answer

    def extract_task(self, text_input: str) -> dict[str, any]:
        task_extraction = {
            "raw_input": text_input,
            "has_command_sign": False,
            "task": None,
            "command": None,
            "obj_list": [],
            "color_list": [],
            "color_map": {},
            "prompt": None
        }
        text_input_ = re.sub("\s+", " ", text_input).strip()
        if len(text_input_) > 0:
            if text_input_[0] == "!":
                task_extraction["has_command_sign"] = True

            command_extraction = re.match(
                pattern=rf"(?i)^\!({self.vision_tasks_commands})",
                string=text_input_
            )
            if command_extraction is not None:
                task = None
                if text_input_ in self.vision_tasks_usage2task.keys():
                    task = self.vision_tasks_usage2task[text_input_]

                if task is not None:
                    if task not in self.vision_tasks_from_text_input_task:
                        task_extraction["task"] = task
                        task_extraction["command"] = self.vision_tasks_task2command[task]
                        task_extraction["prompt"] = task
                    else:
                        return task_extraction

                if task_extraction["task"] is None:
                    command = command_extraction.group()[1:]
                    if command in self.vision_tasks_from_text_input_command:
                        if re.match(
                            pattern=rf"(?i)^\!{command} [^\s]+",
                            string=text_input_
                        ) is not None:
                            task_extraction["task"] = self.vision_tasks_from_text_input_command2task[command]
                            task_extraction["command"] = command
                            task_extraction["obj_list"] = list(set([
                                re.sub('[^A-Za-z]+', ' ', unidecode(obj)) for obj in
                                text_input_.split(" ")[1:]
                                if not any(map(str.isdigit, obj))
                            ]))
                            if len(task_extraction["obj_list"]) > 0:
                                task_extraction["prompt"] = task_extraction["task"] + ", ".join(task_extraction["obj_list"])

        return task_extraction

    def get_history_data(self,
        image: Image.Image,
        hashed_image: str,
        task: str,
        prompt: str,
        command: str,
        obj: str = None
    ) -> dict:
        try:
            if command not in ("detect", "segment"):
                task_res = self.vision_tasks_history[hashed_image][task]
            else:
                task_res = self.vision_tasks_history[hashed_image][task][obj]
        except:
            task_res = None
        
        if task_res is None:
            parsed_answer = self.process_task(
                image=image,
                prompt=prompt,
                task=task
            )
            task_res = parsed_answer[task]
            
            if command not in ("detect", "segment"):
                self.vision_tasks_history[hashed_image][task] = copy.deepcopy(task_res)
            else:
                if command == "detect":
                    task_res = __class__.convert_to_od_format(data=task_res)
                self.vision_tasks_history[hashed_image][task][obj] = copy.deepcopy(task_res)
                
        return task_res

    def load_inputs(self,
        image: Image.Image,
        prompt: str
    ) -> any:
        if image is not None:
            return self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)

    def process_task(self,
            image: Image.Image,
            prompt: str,
            task: str
        ) -> dict:
        inputs = self.load_inputs(image=image, prompt=prompt)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(self.model_dtype),
                max_new_tokens=self.max_new_tokens,
                early_stopping=self.early_stopping,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
            )
        self.clear_cache()

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=self.skip_special_tokens)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )
        
        return parsed_answer

    @staticmethod
    def process_bbox(
        image: Image.Image,
        text_input: str,
        task_res: dict,
        task_extraction: dict,
        vision_task_answer: dict
    ) -> dict:
        command = task_extraction["command"]
        labels = task_res["labels"]
        new_labels = []
        has_answer = False
        
        if len(labels) > 0:
            if command == "ocr":
                bboxes = [[box[i] for i in (0,1,4,5)] for box in task_res["quad_boxes"]]
                labels[0] = labels[0].replace("</s>", "")
            else:
                bboxes = task_res["bboxes"]
            task_extraction["obj_list"] = list(set(labels))
            
            annots = []
            for bbox, label in zip(bboxes, labels):
                if command == "ocr":
                    _, bbox_area = __class__.calculate_overlap(
                        bbox=bbox,
                        image_size=[image.width, image.height]
                    )
                    if bbox_area < 200:
                        continue
                        
                new_labels.append(label)
                annots.append((tuple(map(round, bbox)), label))
                
            if len(new_labels) > 0:
                vision_task_answer = __class__.postprocess(
                    image=image,
                    annots=annots,
                    task_extraction=task_extraction,
                    vision_task_answer=vision_task_answer,
                    labels=new_labels
                )
                has_answer = True

        if not has_answer:
            vision_task_answer["answer"] = f"No result for this task: `{text_input[1:]}`"

        return vision_task_answer
    
    @staticmethod
    def process_mask(
        image: Image.Image,
        polygons: list,
        scale: float = 1,
        mask_opacity: float = 0.75
    ) -> np.ndarray:        
        mask = Image.fromarray(np.zeros(image.size).T)
        draw = ImageDraw.Draw(mask)
        
        for polygon in polygons:
            for _polygon in polygon:
                _polygon = np.array(_polygon).reshape(-1, 2)
                if len(_polygon) < 3:
                    print('Invalid polygon:', _polygon)
                    continue
                
                _polygon = (_polygon * scale).reshape(-1).tolist()
                draw.polygon(_polygon, outline="white", fill="white")
        
        mask = np.array(mask)
        mask = mask_opacity*mask/mask.max()
        
        return mask
    
    @staticmethod
    def postprocess(
        image: Image.Image,
        annots: list,
        task_extraction: dict,
        vision_task_answer: dict,
        labels: list = None
    ) -> dict:
        command = task_extraction["command"]
        
        task_extraction["color_list"] = __class__.get_random_colors(n=len(task_extraction["obj_list"]))
        task_extraction["color_map"] = dict(zip(task_extraction["obj_list"], task_extraction["color_list"]))
        
        vision_task_answer["answer"] = f"Here is the {command} task result !"
        if command in ("ocr"):
            vision_task_answer["answer"] += "\n```\n{ocr_text}\n```".format(ocr_text="\n".join(labels))
        vision_task_answer["is_update_annotated_image"] = True
        vision_task_answer["update_annotated_value"] = (image, annots)
        vision_task_answer["update_annotated_color_map"] = task_extraction["color_map"]
        
        return vision_task_answer

    @staticmethod
    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
        return imports
    
    @staticmethod
    def get_random_colors(n: int) -> list[str]:
        colors = []
        if n > 0:
            while(len(colors) < n):
                color = f"#{random.randint(0, 0xFFFFFF):06x}"
                if color not in colors:
                    colors.append(color)
        
        return colors
    
    @staticmethod
    def convert_to_od_format(data: dict) -> dict:
        bboxes = data.get('bboxes', [])
        labels = data.get('bboxes_labels', [])
        od_results = {
            'bboxes': bboxes,
            'labels': labels
        }

        return od_results
    
    @staticmethod
    def calculate_overlap(bbox: list|tuple, image_size: list|tuple) -> float:
        """Calculates the overlap ratio of a bounding box with an image.

        Args:
            bbox: A list of four integers [x1, y1, x2, y2] representing the bounding box.
            image_size: A tuple of two integers (width, height) representing the image size.

        Returns:
            The overlap ratio as a float between 0 and 1.
        """
        def correct_coordinate(x, max_x):
            if x > max_x:
                x = max_x
            return x

        x1, y1, x2, y2 = [correct_coordinate(x, max_x) for x, max_x in zip(bbox, image_size*2)]
        image_width, image_height = image_size

        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        
        image_area = image_width * image_height

        overlap_ratio = bbox_area / image_area
        
        return overlap_ratio, bbox_area
