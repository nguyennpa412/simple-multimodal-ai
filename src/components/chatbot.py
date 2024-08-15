import os
import gradio as gr

from src.utils.loader import AppLoader
from src.utils.chatbot import ChatBotUtils
from src.components.visual_input import VisualInput

class ChatBot():
    def __init__(self,
        app_loader: AppLoader,
        visual_input: VisualInput
    ) -> None:
        self.app_loader = app_loader
        self.visual_input = visual_input
        self.home_dir = os.environ["APP_HOME_DIR"]
        self.utils = ChatBotUtils(app_loader=self.app_loader, visual_input=self.visual_input)
        
        self.annot_status_icon = gr.Button(
            value="",
            icon=f"{self.home_dir}/assets/icons/annot_gen.gif",
            elem_id="chatbot-annot-status-icon",
            size="lg",
            # min_width=24,
            visible=False
        )
        self.utils.annot_status_icon_url = self.annot_status_icon.icon["url"]

    def get_component(self) -> gr.Group:
        with gr.Column() as chatbot:
            # with gr.Group():
            self.chatbot = gr.Chatbot(
                elem_id="chatbot",
                # show_label=False,
                bubble_full_width=False,
                scale=1,
                avatar_images=(None, self.app_loader.tmp_bot_avatar_path),
                height=511
            )
            
            with gr.Group():
                with gr.Row(elem_id="chatbot-quick-command-row"):
                    self.quick_command_btn_describe = gr.Button(
                        elem_classes="chatbot-quick-command-btn-group",
                        value="Describe",
                        size="sm",
                        interactive=False
                    )
                    self.quick_command_btn_densecap = gr.Button(
                        elem_classes="chatbot-quick-command-btn-group",
                        value="DenseCap",
                        size="sm",
                        interactive=False
                    )
                    self.quick_command_btn_detect = gr.Button(
                        elem_classes="chatbot-quick-command-btn-group",
                        value="Detect",
                        size="sm",
                        interactive=False
                    )
                    self.quick_command_btn_segment = gr.Button(
                        elem_classes="chatbot-quick-command-btn-group",
                        value="Segment",
                        size="sm",
                        interactive=False
                    )
                    self.quick_command_btn_ocr = gr.Button(
                        elem_classes="chatbot-quick-command-btn-group",
                        value="OCR",
                        size="sm",
                        interactive=False
                    )
                    self.quick_command_to_value = {
                        "!describe ": self.quick_command_btn_describe,
                        "!densecap": self.quick_command_btn_densecap,
                        "!detect ": self.quick_command_btn_detect,
                        "!segment ": self.quick_command_btn_segment,
                        "!ocr": self.quick_command_btn_ocr,
                    }
                    self.utils.quick_command_component_num = len(self.quick_command_to_value)
                
                self.chatbot_input = gr.MultimodalTextbox(
                    elem_id="chatbot-input",
                    interactive=False,
                    placeholder=None,
                    show_label=False,
                    scale=1,
                    submit_btn=False,
                    autofocus=False
                )
                
            self.annot_status = gr.Markdown(
                value=None,
                elem_id="chatbot-annot-status",
                visible=False
            )

            self.set_event_listeners()

        return chatbot

    def set_event_listeners(self) -> None:
        self.visual_input.description.change(
            fn=self.utils.update_chatbot_first_interaction,
            inputs=[self.visual_input.input_image, self.visual_input.description],
            outputs=[
                self.chatbot,
                self.chatbot_input
            ] + list(self.quick_command_to_value.values()),
            queue=False
        )
            
        self.quick_command_btn_describe.click(
            fn=lambda: gr.update(value={"text": "!describe "}, autofocus=True),
            inputs=None,
            outputs=[self.chatbot_input],
            queue=False
        )
        self.quick_command_btn_densecap.click(
            fn=lambda: gr.update(value={"text": "!densecap"}, autofocus=True),
            inputs=None,
            outputs=[self.chatbot_input],
            queue=False
        )
        self.quick_command_btn_detect.click(
            fn=lambda: gr.update(value={"text": "!detect "}, autofocus=True),
            inputs=None,
            outputs=[self.chatbot_input],
            queue=False
        )
        self.quick_command_btn_segment.click(
            fn=lambda: gr.update(value={"text": "!segment "}, autofocus=True),
            inputs=None,
            outputs=[self.chatbot_input],
            queue=False
        )
        self.quick_command_btn_ocr.click(
            fn=lambda: gr.update(value={"text": "!ocr"}, autofocus=True),
            inputs=None,
            outputs=[self.chatbot_input],
            queue=False
        )

        self.chatbot_input.focus(
            fn=lambda: gr.update(autofocus=False),
            inputs=None,
            outputs=[self.chatbot_input],
            queue=False
        )

        chat_msg = self.chatbot_input.submit(
            fn=self.utils.add_message,
            inputs=[
                self.visual_input.input_image,
                self.visual_input.description,
                self.chatbot_input,
                self.chatbot
            ],
            outputs=[
                self.chatbot_input,
                self.chatbot,
                self.visual_input.annotated_image,
                self.visual_input.annotated_image_tab,
                self.visual_input.image_tabs,
                self.annot_status,
                self.visual_input.clear_input_image_btn
            ] + list(self.quick_command_to_value.values()),
            queue=False
        )
        bot_msg = chat_msg.then(
            fn=self.utils.respond,
            inputs=[self.visual_input.input_image, self.chatbot],
            outputs=[
                self.chatbot_input,
                self.chatbot
            ] + list(self.quick_command_to_value.values()),
            api_name="bot_response",
            queue=False
        )
        annot_status_update = bot_msg.then(
            fn=self.utils.update_annot_status,
            inputs=None,
            outputs=[self.annot_status],
            queue=False
        )
        annot_img_gen = annot_status_update.then(
            fn=self.utils.update_annotated,
            inputs=None,
            outputs=[
                self.visual_input.annotated_image,
                self.visual_input.annotated_image_tab,
                self.visual_input.image_tabs
            ],
            queue=False
        )
        annot_img_gen.then(
            fn=self.utils.postprocess,
            inputs=None,
            outputs=[self.annot_status, self.visual_input.clear_input_image_btn],
            queue=False
        )
        gr.on(
            triggers=[self.chatbot.change],
            fn=None, inputs=None, outputs=None,
            js='''
                () => {
                    function getAbsoluteHeight(el) {
                        // Get the DOM Node if you pass in a string
                        el = (typeof el === 'string') ? document.querySelector(el) : el; 

                        var styles = window.getComputedStyle(el);
                        var margin = parseFloat(styles['marginTop']) + parseFloat(styles['marginBottom']);

                        return Math.ceil(el.offsetHeight + margin);
                    }

                    let chatbot_log = document.querySelector('#chatbot div[role="log"]');
                    let chatbot_log_messlist = document.querySelector('#chatbot div[role="log"] > div.message-wrap');

                    let near_last_row_height = 0;
                    if (chatbot_log_messlist.children.length > 1) {
                        let near_last_row = chatbot_log_messlist.children[chatbot_log_messlist.children.length - 2];
                        near_last_row_height = getAbsoluteHeight(near_last_row) + 10;
                    }
                    let last_row = chatbot_log_messlist.lastElementChild;
                    let last_row_height = getAbsoluteHeight(last_row);
                    let scroll_to_height = last_row_height + near_last_row_height;
                    
                    let scroll_to = chatbot_log.scrollHeight;

                    if (last_row.classList.contains("bot-row")) {
                        scroll_to = chatbot_log.scrollHeight - scroll_to_height;
                    }

                    try {
                        $('#chatbot div[role="log"]').animate({
                            scrollTop: scroll_to
                        }, 500);
                    } catch(error) {
                        chatbot_log.scrollTop = scroll_to;
                    }
                }
            ''',
            queue=False
        )
        gr.on(
            triggers=[self.visual_input.image_tabs.change],
            fn=None, inputs=None, outputs=None,
            js='''
                () => {
                    try {
                        const legend_children = document.querySelector('#visinp-annotated-image div.legend').childNodes;
                        let legend_dict = {};
                        let legend_remove_list = [];
                        
                        for (let i = 0; i < legend_children.length; i++) {
                            let label_ele = legend_children[i];
                            let label_text = label_ele.textContent;
                            if (label_text in legend_dict) {
                                legend_remove_list.push(label_ele);
                            } else {
                                legend_dict[label_text] = label_ele;
                            }
                        }
                        
                        for (let i = 0; i < legend_remove_list.length; i++) {
                            legend_remove_list[i].remove();
                        }
                        
                    } catch(err) { }
                }
            ''',
            queue=False
        )
