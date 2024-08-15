import os
import re
import time
import copy
import torch
import torchvision.transforms as T
import numpy as np

from typing import Union
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
from torchvision.transforms.functional import InterpolationMode

from src.modules.base_module import BaseModule

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
QUESTION_ATTR = "\npresent as list if possible\nno yapping, short"
LEN_QUESTION_ATTR = len(QUESTION_ATTR)
CHAT_MODES = ("single", "multi", "video")
EMPTY_CHAT = {
    # "idx": 0,
    "history": None,
    "pixel_values": None
}
INIT_CHATS = {
    "single": copy.deepcopy(EMPTY_CHAT),
    "multi": copy.deepcopy(EMPTY_CHAT),
    "video": copy.deepcopy(EMPTY_CHAT),
}
MODEL_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32
}

class VQAChatbot(BaseModule):
    def __init__(self, config: dict) -> None:
        print("\n" + "==||"*20 + "==")
        print("> INIT VQAChatbot...")
        self.set_attributes(**config["load_config"])
        self.set_attributes(**config["load_config"]["specs"][self.specs_type])
        self.warm_up_question = config["warm_up_config"]["question"]
        self.warm_up_image_path = os.environ["APP_HOME_DIR"] + "/" + config["warm_up_config"]["image_path"]
        self.model_dtype = MODEL_DTYPE[self.model_dtype]
        self.use_gpu = True
        self.chat_modes = CHAT_MODES

        self.set_chat_mode(chat_mode=self.chat_modes[0])
        self.clear_chats()
        self.load_device(cuda_only=True)
        self.start_module()

    def set_chat_mode(self, chat_mode: str) -> None:
        self.chat_mode = chat_mode

    def clear_chat(self, chat_mode: str) -> None:
        self.chats[chat_mode] = copy.deepcopy(EMPTY_CHAT)

    def clear_chats(self) -> None:
        self.chats = copy.deepcopy(INIT_CHATS)

    def clear_all(self) -> None:
        self.clear_chats()
        self.clear_model()

    def load_config(self) -> None:
        print("> LOAD CONFIG...")
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path=self.model_hf_path, trust_remote_code=self.trust_remote_code)

        if not self.use_flash_attention:
            self.config.vision_config.update({"use_flash_attn": False})
            self.config.llm_config.update({"attn_implementation": "eager"})

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
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.model_hf_path,
                quantization_config=self.quantization_config,
                torch_dtype=self.model_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                trust_remote_code=self.trust_remote_code,
                config=self.config,
                device_map=self.device
            ).eval()
        else:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.model_hf_path,
                torch_dtype=self.model_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                trust_remote_code=self.trust_remote_code,
                config=self.config,
                device_map=self.device
            ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_hf_path,
            trust_remote_code=self.trust_remote_code
        )
        self.generation_config = dict(
            # num_beams=self.num_beams,
            max_new_tokens=self.max_new_token,
            do_sample=self.do_sample,
        )

    def start_module(self) -> None:
        print("# START MODULE...")
        self.load_config()
        self.load_model()
        print("> WARM UP...")
        print(
            "\n> WARM UP OUTPUT:",
            self.run_model(
                visual_input=self.warm_up_image_path,
                question=self.warm_up_question,
                warm_up=True
            )
        )

    def stop_module(self) -> None:
        print("# STOP MODULE...")
        self.config = None
        self.model = None
        self.tokenizer = None
        self.clear_cache()

    def run_model(self,
        visual_input: Union[str, np.ndarray, list[str], list[np.ndarray]],
        question: str,
        warm_up: bool = False
    ) -> str:
        if (
            (visual_input is not None)\
            & (question is not None)
        ):
            if question.strip() != "":
                self.set_seed(seed=42)
                if self.chats[self.chat_mode]["pixel_values"] is None:
                    self.chats[self.chat_mode]["pixel_values"] = self.load_visual_input(visual_input=visual_input, max_num=self.max_num).to(self.device)
                if self.chats[self.chat_mode]["pixel_values"] is not None:                    
                    print("\n" + "-"*50)
                    print(f"> VQAChatbot: {question}")
                    start_time = time.time()

                    with torch.inference_mode():
                        response, self.chats[self.chat_mode]["history"] = self.model.chat(
                            tokenizer=self.tokenizer,
                            pixel_values=self.chats[self.chat_mode]["pixel_values"],
                            # question=question + QUESTION_ATTR,
                            question=question,
                            generation_config=self.generation_config,
                            history=self.chats[self.chat_mode]["history"],
                            return_history=True,
                        )
                    self.clear_cache()
                    
                    # self.chats[self.chat_mode]["history"][-1] = (
                    #     self.chats[self.chat_mode]["history"][-1][0][:-LEN_QUESTION_ATTR],
                    #     response
                    # )

                    end_time = time.time()
                    print(f"  >> Time elapsed: {end_time - start_time:.2f} s")
                    print("-"*50)
                    
                    self.chats[self.chat_mode]["history"] = self.chats[self.chat_mode]["history"][-self.max_chat_history:]
                    if re.match(r"^<image>\n", self.chats[self.chat_mode]["history"][0][0]) is None:
                        self.chats[self.chat_mode]["history"][0] = (
                            "<image>\n" + self.chats[self.chat_mode]["history"][0][0],
                            self.chats[self.chat_mode]["history"][0][1]
                        )

                    if warm_up:
                        self.clear_chat(chat_mode=self.chat_modes[0])

                    return response

    def load_visual_input(self,
        visual_input: Union[str, np.ndarray, list[str], list[np.ndarray]],
        input_size: int = 448,
        max_num: int = 6
    ) -> torch.Tensor:
        if isinstance(visual_input, (str, np.ndarray)):
            return self.load_single(
                image=visual_input,
                input_size=input_size,
                max_num=max_num
            )
        elif isinstance(visual_input, list):
            return self.load_multi(
                images=visual_input,
                input_size=input_size,
                max_num=max_num
            )

    def load_single(self,
        image: Union[str, np.ndarray],
        input_size: int = 448,
        max_num: int = 6
    ) -> torch.Tensor:
        img = __class__.get_pil_image(image=image)
        if img is not None:
            return __class__.process_image(
                image=img,
                input_size=input_size,
                max_num=max_num
            ).to(self.model_dtype).to(self.device)

    def load_multi(self,
        images: Union[list[str], list[np.ndarray]],
        input_size: int = 448,
        max_num: int = 6
    ) -> torch.Tensor:
        if len(images) > 0:
            if all(isinstance(ele, (str, np.ndarray)) for ele in images):
                imgs = []
                for img in images:
                    imgs.append(
                        __class__.process_image(
                            image=__class__.get_pil_image(image=img),
                            input_size=input_size,
                            max_num=max_num
                        )
                    )
                return torch.cat(imgs, dim=0).to(self.model_dtype).to(self.device)

    @staticmethod
    def process_image(
        image: Image.Image,
        input_size: int = 448,
        max_num: int = 6
    ) -> torch.Tensor:
        img = image.convert('RGB')
        transform = __class__.build_transform(input_size=input_size)
        images = __class__.dynamic_preprocess(
            image=img,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num
        )
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    @staticmethod
    def build_transform(input_size: int) -> T.Compose:
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    @staticmethod
    def find_closest_aspect_ratio(
        aspect_ratio: float,
        target_ratios: list[tuple[int]],
        width: int,
        height: int,
        image_size: int
    ) -> tuple[int]:
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @staticmethod
    def dynamic_preprocess(
        image: Image.Image,
        min_num: int = 1,
        max_num: int = 6,
        image_size: int = 448,
        use_thumbnail: bool = False
    ) -> list[Image.Image]:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = __class__.find_closest_aspect_ratio(
            aspect_ratio=aspect_ratio,
            target_ratios=target_ratios,
            width=orig_width,
            height=orig_height,
            image_size=image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
