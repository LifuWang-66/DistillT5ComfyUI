import comfy.utils
import comfy.sd
import comfy.text_encoders
import folder_paths
from .model import load_clip

class DualCLIPLoaderT5Base:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name1": (folder_paths.get_filename_list("clip"), ),
                              "clip_name2": (folder_paths.get_filename_list("clip"), ),
                              "type": (["flux_t5base"], ),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path_or_raise("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("clip", clip_name2)
        class EmptyClass:
            pass
        clip_target = EmptyClass()
        clip_target.params = {}

        clip = load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (clip,)
    

NODE_CLASS_MAPPINGS = {
    "DualCLIPLoaderT5Base": DualCLIPLoaderT5Base,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualCLIPLoaderT5Base": "DualCLIPLoader for T5Base"
}