import gradio as gr

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as customex_interface:
        with gr.Row():
            with gr.Column():
                with gr.Column(scale=3):
                    gr.Markdown("""<h2><center>ChatGLM Extension</center></h2>""")
                    with gr.Row():
                        load_model = gr.Button("Load ChatGLM Model")
                    
                    with gr.Row():
                        unload_model = gr.Button("Unload ChatGLM Model")

                    with gr.Row():
                        with gr.Column(variant="panel"):
                            with gr.Row():
                                max_length = gr.Slider(minimum=32, maximum=32768, step=32, label='Max Length', value=8192)
                                top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Top P', value=0.8)
                            with gr.Row():
                                temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Temperature', value=0.95)

                            with gr.Row():
                                max_rounds = gr.Slider(minimum=1, maximum=100, step=1, label="Max Round Limit", value=25)
                                apply_max_rounds = gr.Button("✔", elem_id="del-btn")

                            with gr.Row():
                                max_words = gr.Slider(minimum=32, maximum=32768, step=32, label='Max Word Limit', value=8192)
                                apply_max_words = gr.Button("✔", elem_id="del-btn")

                            cmd_output = gr.Textbox(label="Output Message")
                            with gr.Row():
                                use_stream_chat = gr.Checkbox(label='Use Stream Output', value=True)
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            with gr.Row():
                                clear_history_btn = gr.Button("Clear")

                            with gr.Row():
                                sync_his_btn = gr.Button("Synchronize")

                            with gr.Row():
                                save_his_btn = gr.Button("Save")
                                load_his_btn = gr.UploadButton("Load", file_types=['file'], file_count='single')

                            with gr.Row():
                                save_md_btn = gr.Button("Save as MarkDown")

                    with gr.Row():
                        with gr.Column(variant="panel"):
                            with gr.Row():
                                gr.Markdown('''Hint:<br/>`Max Length` Token limit when generating<br/>`Top P` Control the total probability of the top p words in the output text<br/>`Temperature` Control the variety and randomness of the generated text<br/>Smaller `Top P` generates more diverse and irrelevant text, while larger generates more conservative and relevant text<br/>Smaller `Temperature` produces more conservative and relevant text, while larger produces more exotic and irrelevant text.<br/>`Max Round Limit` When the number of session rounds exceeds this value, the earliest session content is discarded<br/>`Max Word Limit` When the number of dialogue words exceeds this number, the earliest dialogue content is discarded<br/>VRAM usage can be reduced by limiting the number of sessions in this way.<br/>Click the dialog to directly modify the dialog content''')

                with gr.Column(scale=7):
                    chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=800)
                    with gr.Row(visible=False) as edit_log:
                        with gr.Column():
                            log = gr.Textbox(placeholder="Enter your modified content", show_label=False, lines=4, elem_id="chat-input").style(container=False)
                            with gr.Row():
                                submit_log = gr.Button('Save')
                                cancel_log = gr.Button('Cancel')
                    log_idx = gr.State([])

                    with gr.Row():
                        input_message = gr.Textbox(placeholder="Enter your words...(Press Ctrl+Enter to send)", show_label=False, lines=4, elem_id="chat-input").style(container=False)
                        clear_input = gr.Button("🗑️", elem_id="del-btn")
                        stop_generate = gr.Button("❌", elem_id="del-btn")

                    with gr.Row():
                        submit = gr.Button("Send", elem_id="c_generate")

                    with gr.Row():
                        revoke_btn = gr.Button("Revoke")
                    
                    with gr.Row():
                        regenerate_btn = gr.Button("Regenerate")
    return [(customex_interface, 'ChatGLM', 'chatglm')]