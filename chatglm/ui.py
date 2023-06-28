import gradio as gr

from chatglm.model import infer
from chatglm.context import Context
from chatglm.model import load_model
from chatglm.model import unload_model

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

def predict(ctx, query, max_length, top_p, temperature, use_stream_chat):
    ctx.limit_round()
    ctx.limit_word()

    ctx.inferBegin()
    token = 0
    ctx_round = ctx.get_round()
    ctx_word = ctx.get_word()
    yield ctx.rh, "Generating...", f"Round: {ctx_round}\nWord: {ctx_word}\nToken: {token}"

    for _, output in infer(
            query=query,
            history=ctx.history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            use_stream_chat=use_stream_chat
    ):
        if ctx.inferLoop(query, output):
            print("")
            break

        token += 1
        yield ctx.rh, gr_show(), f"Round: {ctx_round}\nWord: {ctx_word}\nToken: {token}"

    ctx.inferEnd()
    yield ctx.rh, "", f"Round: {ctx_round}\nWord: {ctx_word}\nLast Token: {token}"

def regenerate(ctx, max_length, top_p, temperature, use_stream_chat):
    if not ctx.rh:
        print('*' * 50)
        raise RuntimeError("Content does not exist")
    
    query, output = ctx.rh.pop()
    ctx.history.pop()

    for p0, p1, p2 in predict(ctx, query, max_length, top_p, temperature, use_stream_chat):
        yield p0, p1, p2

def clear_history(ctx):
    ctx.clear()
    return gr.update(value=[]), "Successfully cleared all Contents"

def edit_history(ctx, log, idx):
    if log == '':
        return ctx.rh, {'visible': True, '__type__': 'update'},  {'value': ctx.history[idx[0]][idx[1]], '__type__': 'update'}, idx
    print('+' * 50)
    print(ctx.history[idx[0]][idx[1]])
    print("----->")
    print(log)
    ctx.edit_history(log, idx[0], idx[1])
    return ctx.rh, *gr_hide()

def gr_show_and_load(ctx, evt: gr.SelectData):
    if evt.index[1] == 0:
        label = f'Modify Q{evt.index[0]}:'
    else:
        label = f'Modify A{evt.index[0]}:'
    return {'visible': True, '__type__': 'update'}, {'value': ctx.history[evt.index[0]][evt.index[1]], 'label': label, '__type__': 'update'}, evt.index

def gr_hide():
    return {'visible': False, '__type__': 'update'}, {'value': '', 'label': '', '__type__': 'update'}, []

def apply_max_round_click(ctx, max_round):
    ctx.max_rounds = max_round
    return f"Applied: Max Round Limit {ctx.max_rounds}"

def apply_max_words_click(ctx, max_words):
    ctx.max_words = max_words
    return f"Applied: Max Word Limit {ctx.max_words}"

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as customex_interface:
        _ctx = Context()
        state = gr.State(_ctx)
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""<h2><center>ChatGLM Extension</center></h2>""")
                with gr.Row():
                    loadmodel = gr.Button("Load ChatGLM Model")
                
                with gr.Row():
                    unloadmodel = gr.Button("Unload ChatGLM Model")

                with gr.Row():
                    precision = gr.Radio(choices=["fp32", "bf16", "fp16", "int8", "int4"], value="fp16", label="Precision", elem_id="checkpoint_precision")

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

            
            loadmodel.click(load_model, inputs=[precision], outputs=[cmd_output])
            unloadmodel.click(unload_model, inputs=[], outputs=[cmd_output])

            submit.click(predict, inputs=[
                state,
                input_message,
                max_length,
                top_p,
                temperature,
                use_stream_chat
            ], outputs=[
                chatbot,
                input_message,
                cmd_output
            ])

            regenerate_btn.click(regenerate, inputs=[
                state,
                max_length,
                top_p,
                temperature,
                use_stream_chat
            ], outputs=[
                chatbot,
                input_message,
                cmd_output
            ])
            
            revoke_btn.click(lambda ctx: ctx.revoke(), inputs=[state], outputs=[chatbot])
            clear_history_btn.click(clear_history, inputs=[state], outputs=[chatbot, cmd_output])
            stop_generate.click(lambda ctx: ctx.interrupt(), inputs=[state], outputs=[])
            clear_input.click(lambda x: "", inputs=[input_message], outputs=[input_message])
            save_his_btn.click(lambda ctx: ctx.save_history(), inputs=[state], outputs=[cmd_output])
            save_md_btn.click(lambda ctx: ctx.save_as_md(), inputs=[state], outputs=[cmd_output])
            load_his_btn.upload(lambda ctx, f: ctx.load_history(f), inputs=[state, load_his_btn], outputs=[chatbot])
            sync_his_btn.click(lambda ctx: ctx.rh, inputs=[state], outputs=[chatbot])
            apply_max_rounds.click(apply_max_round_click, inputs=[state, max_rounds], outputs=[cmd_output])
            apply_max_words.click(apply_max_words_click, inputs=[state, max_words], outputs=[cmd_output])
            chatbot.select(gr_show_and_load, inputs=[state], outputs=[edit_log, log, log_idx])
            submit_log.click(edit_history, inputs=[state, log, log_idx], outputs=[chatbot, edit_log, log, log_idx])
            cancel_log.click(gr_hide, outputs=[edit_log, log, log_idx])
    
    return [(customex_interface, 'ChatGLM', 'chatglm')]