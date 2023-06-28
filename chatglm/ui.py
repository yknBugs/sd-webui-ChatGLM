import gradio as gr

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as customex_interface:
        with gr.Row():
            with gr.Column():
                gr.Markdown("ChatGLM")
    return [(customex_interface, 'ChatGLM', 'chatglm')]