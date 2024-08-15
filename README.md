# Simple Application for Multimodal AI

## Table of Contents
1. [Introduction](#1-introduction)
2. [Main features](#2-main-features)
3. [Demos](#3-demos)
4. [Installation](#4-installation)
5. [Configs & Run](#5-configs--run)
6. [Docker](#6-docker)
7. [References](#references)

## 1. Introduction

A simple yet vesatile application using Gradio, featuring the integration of various open-source models from Hugging Face. This app supports a range of tasks, including Image Text to Text, Visual Question Answering, and Text to Speech, providing an accessible interface for experimenting with these advanced machine learning models.

![cover](https://github.com/user-attachments/assets/17b245dd-29eb-4596-9c8b-012ace0bcd3b)

## 2. Main features

<table>
    <thead>
        <tr>
            <th scope="col"></th>
            <th scope="col">Module</th>
            <th scope="col">Source</th>
            <th scope="col">Function</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th scope="row">#M1</th>
            <td><b>Image-Text-to-Text<b></td>
            <td><a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large</a></td>
            <td>
                <ul>
                    <li><i>Description Generation</i></li>
                    <li><i>Computer Vision Tasks</i></li>
                </ul>
            </td>
        </tr>
        <tr>
            <th scope="row">#M2</th>
            <td><b>Visual Question Answering</b></td>
            <td><a href="https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5">OpenGVLab/Mini-InternVL-Chat-2B-V1-5</a></td>
            <td>
                <ul>
                    <li><i>Chatbot</i></li>
                </ul>
            </td>
        </tr>
        <tr>
            <th scope="row">#M3</th>
            <td><b>Text-to-Speech</b></td>
            <td><a href="https://huggingface.co/coqui/XTTS-v2">coqui/XTTS-v2</a></td>
            <td>
                <ul>
                    <li><i>Description Speech Generation</i></li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

<details>
    <summary><i><b>Computer Vision Tasks details</b></i></summary>
    <table>
        <thead>
            <tr>
                <th>Task type</th>
                <th>Task details</th>
                <th>Usage</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan=4><b>Image Captioning</b></td>
                <td><i>Generate a short description</i></td>
                <td><code>!describe -s</code></td>
            </tr>
            <tr>
                <td><i>Generate a detailed description</i></td>
                <td><code>!describe -m</code></td>
            </tr>
            <tr>
                <td><i>Generate a more detailed description</i></td>
                <td><code>!describe -l</code></td>
            </tr>
            <tr>
                <td><i>Localize and Describe salient regions</i></td>
                <td><code>!densecap</code></td>
            </tr>
            <tr>
                <td><b>Object Detection</b></td>
                <td><i>Detect objects from text inputs</i></td>
                <td><code>!detect obj1 obj2 ...</code></td>
            </tr>
            <tr>
                <td><b>Image Segmentation</b></td>
                <td><i>Segment objects from text inputs</i></td>
                <td><code>!segment obj1 obj2 ...</code></td>
            </tr>
            <tr>
                <td><b>Optical Character Recognition</b></td>
                <td><i>Localize and Recognize text</i></td>
                <td><code>!ocr</code></td>
            </tr>
        </tbody>
    </table>
</details>

>### Additional features
> - **Voice options**: _You can choose the voice for Speech Synthesizer, there are currently 2 voice options:_
>     - David Attenborough
>     - Morgan Freeman
> - **Random bot**: _With every input image entry, a different random bot avatar would be used._ 
>     - <details><summary>Demo</summary><video src="https://github.com/user-attachments/assets/4fcf8614-a942-4b90-92fe-b9c98ee0165d"></video></details>

## 3. Demos
<details>
    <summary><b>Image-Text-to-Text</b></summary>
    <video src="https://github.com/user-attachments/assets/22d2950b-2466-4780-b96b-913f06b1a5d2"></video>
</details>

<details>
    <summary><b>Visual Question Answering</b></summary>
    <video src="https://github.com/user-attachments/assets/7b9d05a0-df7a-408e-b509-483dd51c9565"></video>
</details>

<details>
    <summary><b>Text-to-Speech</b></summary>
    <video src="https://github.com/user-attachments/assets/becea701-2328-47f8-8eed-c4ae4cacd1e6"></video>
</details>

## 4. Installation
### 4.1 Tested environment
- Ubuntu `22.04`
- Python `3.10.12`
- NVIDIA driver `555`
- CUDA `11.8`
- CuDNN`8` & CuDNN`9`

### 4.2 GPU requirements
- **Capable of processing on GPU and CPU**:

|   | GPU | CPU |
|:-:|:---:|:---:|
| **#M1** | ✅ | ✅ |
| **#M2** | ✅ | ❌ |
| **#M3** | ✅ | ✅ |

> - _Do you need GPU to run this app?_
>     - _No, you can run this app on CPU_, but you can only use `Image-Text-to-Text` and `Text-to-Speech` modules, also processing time would be longer.

- **GPU consumptions**:

![GPU_consumptions](https://github.com/user-attachments/assets/e9b0c14b-e26d-4730-a7a3-7d67c79d8021)

> - You can set `dtype` and `quantization` based on this table so that you can make full use of your GPU.
> - For example with my **6GB GPU**:
>     - **#M1**: `gpu - q4 - bfp16`
>     - **#M2**: `gpu - q8 - bfp16`
>     - **#M3**: `cpu - fp32`
>         - This is the current `gpu_low` specs config.

### 4.3 Installation
> This preparation is for local run, you should use a `venv` for local run.
- **CPU only**: Run `pip install -r requirements.cpu.txt`
- **GPU**:
    - Install suitable NVIDIA driver
    - Install CUDA `11.8` & CuDNN`8|9`
    - `pip install -r requirements.txt`

## 5. Configs & Run
### 5.1 Config files
<table>
    <thead>
        <tr>
            <th scope="col"></th>
            <th scope="col">File</th>
            <th scope="col">Includes</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>General configs<b></td>
            <td><a href="./configs/app_config.yaml">app_config.yaml</a></td>
            <td>
                <ul>
                    <li><i>Module configs</i></li>
                    <li><i>App configs</i></li>
                    <li><i>Launch configs</i></li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><b>#M1 configs</b></td>
            <td><a href="./configs/image_text_to_text/florence_2_large.yaml">florence_2_large.yaml</a></td>
            <td rowspan=3>
                <ul>
                    <li><i>Load configs</i></li>
                    <li><i>Warm-up configs</i></li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><b>#M2 configs</b></td>
            <td><a href="./configs/visual_question_answering/mini_internvl_chat_2b_v1_5.yaml">mini_internvl_chat_2b_v1_5.yaml</a></td>
        </tr>
        <tr>
            <td><b>#M3 configs</b></td>
            <td><a href="./configs/text_to_speech/xtts_v2.yaml">xtts_v2.yaml</a></td>
        </tr>
    </tbody>
</table>

### 5.2 Specs configs

There are 3 profiles for specs configs:
| | cpu | gpu_low | gpu_high |
|:-:|:---:|:-------:|:--------:|
| **#M1** | `cpu - fp32` | `gpu - q4 - bfp16` | `gpu - fp32` |
| **#M2** | | `gpu - q8 - bfp16` | `gpu - fp32` |
| **#M3** | `cpu - fp32` | `cpu - fp32` | `gpu - fp32` |
| **GPU VRAM needed** | 0 | ~6GB | > 16GB |

>- With `gpu_high`, **#M3** will use longer speaker voice duration for synthesizing.
>- The ***current default profile*** is `gpu_low`. You can set the specs profile in [app_config.yaml](./configs/app_config.yaml).
>- If you want to create a ***custom profile*** for this, remember to add the custom profile to all module config files as well.

### 5.3 Run the app (Local)

- **Share option**: To create a ***temporary shareable link*** for others to use the app, simply set `share` -> `True` under `lanch_config` in [app_config.yaml](./configs/app_config.yaml) before running the app.
- **Run the app**:
    - Activate `venv` (Optional)
    - `python app.py`

> The app is running on http://127.0.0.1:7860/

## 6. Docker

### 6.1 NVIDIA Container Toolkit

> You need to install `NVIDIA Container Toolkit` in order to use docker for gpu images.
- [Installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Checking after installation](https://docs.docker.com/engine/containers/resource_constraints/#expose-gpus-for-use)

### 6.2 Build new images

> Remember to change the ***specs profile*** in [app_config.yaml](./configs/app_config.yaml) before building images.

- **Docker engine build**:
    - **CPU specs**: `docker build -f Dockerfile.cpu -t {image_name}:{tag} .`
    - **GPU specs**: `docker build -t {image_name}:{tag} .`

- **Docker compose build**:
    - **CPU specs**:
        - Change `image` in [docker-compose.cpu.yaml](./docker-compose.cpu.yaml) to your liking
        - `docker compose -f docker-compose.cpu.yaml build`
    - **GPU specs**:
        - Change `image` in [docker-compose.yaml](./docker-compose.yaml) to your liking
        - `docker compose build`

### 6.3 Run built images

- **Docker engine run**:
    - **CPU image**: `docker run -p 7860:7860 {image_name}:{tag}`
    - **GPU image**: `docker run --gpus all -p 7860:7860 {image_name}:{tag}`

- **Docker compose run**:
    - **CPU image**: `docker compose -f docker-compose.cpu.yaml up`
    - **GPU image**: `docker compose up`

> The app is running on http://0.0.0.0:7860/

### 6.4 Pre-built images

> - Docker Hub repository: https://hub.docker.com/r/nguyennpa412/simple-multimodal-ai
> - There are 3 tags for 3 specs profiles: `cpu`, `gpu-low`, `gpu-high`

- **Docker engine run**:
    - **cpu**: `docker run --pull=always -p 7860:7860 nguyennpa412/simple-multimodal-ai:cpu`
    - **gpu-low**: `docker run --pull=always --gpus all -p 7860:7860 nguyennpa412/simple-multimodal-ai:gpu-low`
    - **gpu-high**: `docker run --pull=always --gpus all -p 7860:7860 nguyennpa412/simple-multimodal-ai:gpu-high`

- **Docker compose run**:
    - **cpu**:
        - Change `image` in [docker-compose.cpu.yaml](./docker-compose.cpu.yaml) to `nguyennpa412/simple-multimodal-ai:cpu`
        - `docker compose -f docker-compose.cpu.yaml up --pull=always`
    - **gpu-low**:
        - Change `image` in [docker-compose.yaml](./docker-compose.yaml) to `nguyennpa412/simple-multimodal-ai:gpu-low`
        - `docker compose up --pull=always`
    - **gpu-high**:
        - Change `image` in [docker-compose.yaml](./docker-compose.yaml) to `nguyennpa412/simple-multimodal-ai:gpu-high`
        - `docker compose up --pull=always`

> The app is running on http://0.0.0.0:7860/

## References
1. B. Xiao *et al.*, "Florence-2: Advancing a unified representation for a variety of vision tasks," arXiv preprint arXiv:2311.06242, 2023. [Online]. Available: https://arxiv.org/abs/2311.06242
2. Z. Chen *et al.*, "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks," arXiv preprint arXiv:2312.14238, 2023. [Online]. Available: https://arxiv.org/abs/2312.14238
3. Z. Chen *et al.*, "How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites," arXiv preprint arXiv:2404.16821, 2024. [Online]. Available: https://arxiv.org/abs/2404.16821
4. E. Casanova *et al.*, "XTTS: A Massively Multilingual Zero-Shot Text-to-Speech Model," arXiv preprint arXiv:2406.04904, 2024. [Online]. Available: https://arxiv.org/abs/2406.04904
