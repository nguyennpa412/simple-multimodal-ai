import os
import gradio as gr

from src.utils.loader import AppLoader
from src.components.chatbot import ChatBot
from src.components.visual_input import VisualInput

class Main():
    def __init__(self, app_loader: AppLoader) -> None:
        self.app_loader = app_loader
        self.home_dir = os.environ["APP_HOME_DIR"]
        self.visual_input = VisualInput(app_loader=self.app_loader)
        self.chatbot = ChatBot(app_loader=self.app_loader, visual_input=self.visual_input)

    def get_config_panel(self) -> gr.Accordion:
        with gr.Row(equal_height=False) as config_panel:
            with gr.Column(scale=7):
                with gr.Accordion(
                    label="Task - Model info",
                    open=False,
                    elem_id="tasmod-info"
                ) as self.tasmod_info:
                    for module in self.app_loader.modules.keys():
                        with gr.Tab(
                            label=self.app_loader.modules[module].task_name,
                            elem_classes="tasmod-info-tab"
                        ):
                            gr.Markdown(
                                value="- **Source**: [{model}](https://huggingface.co/{model})".format(
                                    model=self.app_loader.modules[module].model_hf_path
                                )
                            )
                            gr.Markdown(
                                value="- **Functions**: *{functions}*".format(
                                    functions=self.app_loader.modules[module].functions
                                )
                            )

            with gr.Column(scale=3):
                with gr.Accordion(
                    label="Computer Vision Tasks info",
                    open=False,
                    elem_id="vistas-info"
                ) as self.vistas_info:
                    gr.Markdown(
                        value="💡 **Functions** & **Commands**",
                        elem_id="vistas-tooltip"
                    )

                    df = self.app_loader.vision_tasks_config
                    task_types = tuple(df["type"].unique())
                    with gr.Group():
                        for task_type in task_types:
                            with gr.Accordion(
                                label=task_type,
                                open=False
                            ):
                                df_ = df[df["type"] == task_type]
                                for row in df_.iloc:
                                    gr.Markdown(
                                        value="""
                                        - {description}
                                            > `{usage}`
                                        """.format(
                                            description=row["description"],
                                            usage=row["usage"]
                                        )
                                    )
                                    if len(row["sub_info"]) > 0:
                                        sub_info_list = "\n".join([f"<li class='no-margin'><em>{info}</em></li>" for info in row["sub_info"]])
                                        gr.Markdown(f"<ul><ul class='margin-top-1-5'>\n{sub_info_list}\n</ul></ul>")

        return config_panel

    def get_component(self) -> gr.Blocks:
        with gr.Blocks(
            title=self.app_loader.title,
            theme=self.app_loader.theme,
            css=self.app_loader.css_path,
            head='<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>',
            js='''
                () => {
                    const textbox = document.querySelector("#chatbot-input textarea");
                    
                    textbox.addEventListener("drop", (event) => {
                        event.preventDefault();
                    });
                    
                    document.addEventListener('animationstart', function(event){
                        try {
                            const thumbnails = document.querySelector("#chatbot-input label.container > div.thumbnails");
                            thumbnails.remove();
                        } catch(err) { }
                    }, true);
                }
            '''
        ) as main:
            # TITLE
            self.title = gr.Markdown(value=f"# {self.app_loader.title}")

            with gr.Tab(label="ImageQA+"):
                self.config_panel = self.get_config_panel()

                with gr.Row():
                    with gr.Column(scale=7):
                        self.visual_input_component = self.visual_input.get_component()

                    with gr.Column(scale=3):
                        self.chatbot_component = self.chatbot.get_component()

        return main