import comfy.utils
import comfy.sd
import comfy.text_encoders
import torch
from comfy import sd1_clip
import os
import logging
from .t5 import T5Base

def load_clip(ckpt_paths, embedding_directory=None, model_options={}):
    clip_data = []
    for p in ckpt_paths:
        clip_data.append(comfy.utils.load_torch_file(p, safe_load=True))
    return load_text_encoder_state_dicts(clip_data, embedding_directory=embedding_directory, model_options=model_options)


def load_text_encoder_state_dicts(state_dicts=[], embedding_directory=None, model_options={}):
    clip_data = state_dicts

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = comfy.utils.clip_text_transformers_convert(clip_data[i], "", "")
        else:
            if "text_projection" in clip_data[i]:
                clip_data[i]["text_projection.weight"] = clip_data[i]["text_projection"].transpose(0, 1) #old models saved with the CLIPSave node

    clip_target = EmptyClass()
    clip_target.params = {}
    
    if len(clip_data) == 2:
        clip_target.clip = FluxClipModel
        clip_target.tokenizer = comfy.text_encoders.flux.FluxTokenizer


    parameters = 0
    tokenizer_data = {}
    for c in clip_data:
        parameters += comfy.utils.calculate_parameters(c)
        tokenizer_data, model_options = comfy.text_encoders.long_clipl.model_options_long_clip(c, tokenizer_data, model_options)

    clip = comfy.sd.CLIP(clip_target, embedding_directory=embedding_directory, parameters=parameters, tokenizer_data=tokenizer_data, model_options=model_options)
    for c in clip_data:
        m, u = clip.load_sd(c)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))
    return clip


class FluxClipModel(torch.nn.Module):
    def __init__(self, dtype_t5=None, device="cpu", dtype=None, model_options={}):
        super().__init__()
        dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
        clip_l_class = model_options.get("clip_l_class", sd1_clip.SDClipModel)
        self.clip_l = clip_l_class(device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options)
        self.t5base = T5Model(device=device, dtype=dtype_t5, model_options=model_options)
        self.dtypes = set([dtype, dtype_t5])

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.t5base.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.t5base.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]

        t5_out, t5_pooled = self.t5base.encode_token_weights(token_weight_pairs_t5)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return t5_out, l_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            new_state_dict = {}
            for key, value in sd.items():
                if "shared.weight" in key:
                    new_state_dict["shared.weight"] = value
                elif key.startswith("encoder.encoder."):
                    new_key = key.replace("encoder.encoder.", "encoder.", 1)
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value 
            sd = new_state_dict
            return self.t5base.load_sd(sd)

class T5Model(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=False, model_options={}):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_config_base_projection.json")
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=T5Base, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)