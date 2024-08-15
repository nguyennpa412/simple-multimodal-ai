import gc
import torch
import random
import numpy as np

from PIL import Image
from typing import Union
from hashlib import sha256

class BaseModule():
    def set_attributes(self, **kwargs) -> None:
        for key in kwargs:
            self.__setattr__(key, kwargs[key])

    def set_device(self, device: str = "cpu") -> None:
        self.device = device
        if self.device == "cpu":
            self.use_gpu = False
        else:
            self.use_gpu = True

    def load_device(self, cuda_only: bool = False) -> None:
        device = "cpu"
        if self.use_gpu:
            if torch.cuda.is_available():
                device = "cuda:0"
                print(">> Using GPU...")
            else:
                print(">> No CUDA available, using CPU instead...")
        else:
            print(">> Using CPU...")

        assert not (cuda_only & (device == "cpu")), "GPU-only model!"

        self.set_device(device)

    def set_seed(self, seed: int = 42) -> None:
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

    def load_config(self, **kwargs: any) -> any:
        pass

    def load_model(self, **kwargs: any) -> any:
        pass

    def run_model(self, **kwargs: any) -> any:
        pass

    def clear_model(self) -> None:
        self.model = None
        self.clear_cache()

    def clear_cache(self) -> None:
        gc.collect()
        if self.use_gpu:
            torch.cuda.empty_cache()
        
    def __call__(self, **kwargs: any) -> any:
        self.run_model(**kwargs)

    @staticmethod
    def hash_text(text: str) -> str:
        return sha256(text.encode("utf8")).hexdigest()
    
    @staticmethod
    def get_pil_image(image: Union[str, np.ndarray]) -> Image.Image:
        img = None
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        return img
