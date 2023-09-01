from typing import Optional, List, Tuple

import os
import traceback
import torch
import modules.scripts as scripts

tokenizer = None
model = None
model_path = os.path.join(scripts.basedir(), "model")


def prepare_model(precision):
    global model
    if precision == "fp32":
        model = model.float()
    elif precision == "bf16":
        model = model.bfloat16()
    elif precision == "fp16":
        model = model.half().cuda()
    elif precision == "int8":
            model = model.half().quantize(8).cuda()
    elif precision == "int4":
        model = model.half().quantize(4).cuda()

    model = model.eval()


def load_model(precision):
    from transformers import AutoModel, AutoTokenizer # , AutoConfig
    # import torch

    global tokenizer, model

    # # Load pretrained model and tokenizer
    # config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # config.pre_seq_len = 128
    # config.prefix_projection = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)

    # if cmd_opts.ptuning_checkpoint is not None:
    #     # Load ptuning weights
    #     prefix_state_dict = torch.load(os.path.join(cmd_opts.ptuning_checkpoint, "pytorch_model.bin"))
    #     new_prefix_state_dict = {}
    #     for k, v in prefix_state_dict.items():
    #         if k.startswith("transformer.prefix_encoder."):
    #             new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    #
    #    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    prepare_model(precision)
    return f"Successfully load model using {precision}"

def unload_model():
    global model
    if model is not None:
        del model
        model = None
        if torch.cuda.is_available():
            with torch.cuda.device("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        return "Successfully unload model"
    return "Model not load"

def infer(query,
          history: Optional[List[Tuple]],
          max_length, top_p, temperature, use_stream_chat: bool):
    if not model:
        raise RuntimeError("Model not load")

    if history is None:
        history = []

    output_pos = 0
    try:
        print(query)
        if use_stream_chat:
            for output, history in model.stream_chat(
                    tokenizer, query=query, history=history,
                    max_length=max_length,
                    top_p=top_p,
                    temperature=temperature
            ):
                print(output[output_pos:], end='', flush=True)
                output_pos = len(output)
                yield query, output

        else:
            output, history = model.chat(
                tokenizer, query=query, history=history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature
            )

            print(output, end='')
            yield query, output

    except Exception as e:
        traceback.print_exc()

    print()
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()