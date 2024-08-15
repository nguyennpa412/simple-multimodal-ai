import os
import gradio as gr

from src.utils.loader import AppLoader
from src.utils.visual_input import VisualInputUtils

class VisualInput():
    def __init__(self, app_loader: AppLoader) -> None:
        self.app_loader = app_loader
        self.home_dir = os.environ["APP_HOME_DIR"]
        self.demo_image_path = self.app_loader.demo_image_path
        self.demo_images_path_list = self.app_loader.demo_images_path_list
        self.utils = VisualInputUtils(app_loader=self.app_loader)
        self.app_loader.visinp_demo_images_interactive = True
        self.app_loader.visinp_selected_demo_image_idx = -1
        try:
            self.speaker_dropdown_choices = [
                (speaker.replace("_", " ").title(), speaker)
                for speaker in self.app_loader.modules["tts"].speakers
            ]
        except:
            self.speaker_dropdown_choices = [("David Attenborough", "david_attenborough")]

    def get_component(self) -> gr.Group:
        with gr.Row() as visual_input:
            with gr.Column(scale=0, min_width=160):
                self.demo_images = gr.Gallery(
                    value=self.demo_images_path_list,
                    selected_index=None,
                    elem_id="visinp-demo-images",
                    label="Demo Images",
                    columns=1,
                    show_download_button=False,
                    show_share_button=False,
                    object_fit="contain",
                    allow_preview=True,
                    height=621,
                    interactive=False
                )

            with gr.Column(scale=1):
                with gr.Tabs(elem_id="visinp-image-tabs") as self.image_tabs:
                    with gr.Tab(
                        label="Input Image",
                        elem_id="visinp-input-image-tab",
                        id=0
                    ) as self.input_image_tab:
                        self.input_image = gr.Image(
                            label="Input Image",
                            show_label=False,
                            elem_id="visinp-input-image",
                            show_download_button=False,
                            show_share_button=False
                        )
                    with gr.Tab(
                        label="Annotated Image",
                        elem_id="visinp-annotated-image-tab",
                        id=1,
                        visible=False
                    ) as self.annotated_image_tab:
                        self.annotated_image = gr.AnnotatedImage(
                            label="Annotated Image",
                            show_label=False,
                            elem_id="visinp-annotated-image",
                            visible=False
                        )

                with gr.Row():
                    self.description = gr.Textbox(
                        label="Description",
                        elem_id="visinp-description",
                        interactive=False,
                        scale=1,
                        # lines=1
                    )
                    self.description_speech_btn = gr.Button(
                        value="",
                        icon=f"{self.home_dir}/assets/icons/audio_play.png",
                        elem_id="visinp-description-speech-btn",
                        size="sm",
                        scale=0,
                        min_width=66
                    )

                with gr.Group():
                    with gr.Row():
                        self.description_speech_audio = gr.Audio(
                            show_label=False,
                            elem_id="visinp-description-speech-audio",
                            interactive=False,
                            autoplay=True,
                            visible=True,
                            show_download_button=False,
                            show_share_button=False,
                            waveform_options=gr.WaveformOptions(show_controls=False),
                            scale=1,
                        )
                        self.description_speaker_avatar = gr.Image(
                            value=self.app_loader.bot_avatar,
                            elem_id="visinp-description-speaker-avatar",
                            show_label=False,
                            show_download_button=False,
                            show_share_button=False,
                            interactive=False,
                            mirror_webcam=False,
                            scale=0,
                            min_width=66,
                            height=112
                        )

                with gr.Row():
                    self.clear_input_image_btn = gr.ClearButton(
                        components=self.input_image,
                        elem_id="visinp-clear-input-image-btn",
                        value="Clear Image",
                        size="sm",
                        interactive=False
                    )
                    self.speaker_dropdown = gr.Dropdown(
                        choices=self.speaker_dropdown_choices,
                        elem_id="visinp-speaker-dropdown",
                        label="Voice",
                        type="index",
                        value=0,
                        multiselect=False,
                        allow_custom_value=False,
                        filterable=False
                    )

            self.set_event_listeners()

        return visual_input

    def set_event_listeners(self) -> None:
        self.demo_images.select(
            fn=self.utils.select_demo_image,
            inputs=[self.input_image, self.demo_images],
            outputs=[
                self.input_image,
                self.demo_images,
                self.description
            ]
        )
        gr.on(
            triggers=[self.demo_images.select],
            fn=None, inputs=None, outputs=None,
            js='''
                () => {
                    var clear_btn = document.querySelector('#visinp-demo-images button[title="Clear"]');
                    clear_btn.click();
                }
            '''
        )

        self.input_image.change(
            fn=self.utils.get_description,
            inputs=[self.input_image],
            outputs=[self.description],
            queue=False
        )
        self.input_image.change(
            fn=self.utils.input_image_handle,
            inputs=[self.input_image],
            outputs=[
                self.clear_input_image_btn,
                self.annotated_image,
                self.annotated_image_tab,
                self.image_tabs
            ],
            queue=False
        )

        self.description.change(
            fn=self.utils.description_handle,
            inputs=[self.description, self.description_speech_audio],
            outputs=[self.description_speech_audio, self.clear_input_image_btn],
            queue=False
        )

        self.speaker_dropdown.select(
            fn=self.utils.select_speaker,
            inputs=None,
            outputs=None,
            queue=False
        )

        self.description_speech_btn.click(
            fn=self.utils.get_speech,
            inputs=[self.description, self.description_speech_audio],
            outputs=[self.description_speech_audio],
            queue=False,
            show_progress=True
        )
        self.description_speech_btn.click(
            fn=self.utils.synth_handle,
            inputs=[
                self.description,
                self.description_speech_audio,
                self.description_speech_btn,
                self.clear_input_image_btn
            ],
            outputs=[self.description_speech_btn, self.clear_input_image_btn],
            queue=False
        )
        self.description_speech_audio.play(
            fn=self.utils.release_interactive,
            inputs=None,
            outputs=[self.clear_input_image_btn],
            queue=False
        )
        self.description_speech_audio.stop(
            fn=self.utils.get_speech,
            inputs=[self.description, self.description_speech_audio],
            outputs=[self.description_speech_audio],
            queue=False
        )
        self.description_speech_audio.change(
            fn=self.utils.change_speech_icon,
            inputs=[self.description, self.description_speech_audio],
            outputs=[
                self.description_speech_btn,
                self.description_speaker_avatar,
                self.clear_input_image_btn
            ],
            queue=False
        )
