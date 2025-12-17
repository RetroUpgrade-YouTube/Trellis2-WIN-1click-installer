import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageSequence, ImageOps
from pathlib import Path
import numpy as np
import json
import trimesh as Trimesh
from tqdm import tqdm

import folder_paths
import node_helpers
import hashlib

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.utils

from .trellis2.pipelines import Trellis2ImageTo3DPipeline

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
    
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
def convert_tensor_images_to_pil(images):
    pil_array = []
    
    for image in images:
        pil_array.append(tensor2pil(image))
        
    return pil_array

class Trellis2LoadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelname": (["TRELLIS.2-4B"],),
                "backend": (["flash_attn","xformers"],{"default":"xformers"}),
            },
        }

    RETURN_TYPES = ("TRELLIS2PIPELINE", )
    RETURN_NAMES = ("pipeline", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, modelname, backend):
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
        os.environ['ATTN_BACKEND'] = backend
        
        download_path = os.path.join(folder_paths.models_dir,"microsoft")
        model_path = os.path.join(download_path, modelname)
        
        hf_model_name = f"microsoft/{modelname}"
        
        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=hf_model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
            
        dinov3_model_path = os.path.join(folder_paths.models_dir,"facebook","dinov3-vitl16-pretrain-lvd1689m","model.safetensors")
        if not os.path.exists(dinov3_model_path):
            raise Exception("Facebook Dinov3 model not found in models/facebook/dinov3-vitl16-pretrain-lvd1689m folder")
        
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        pipeline.cuda()
        
        return (pipeline,)
        
class Trellis2MeshWithVoxelGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "pipeline_type": (["512","1024","1024_cascade","1536_cascade"],{"default":"1024_cascade"}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "texture_steps": ("INT",{"default":12, "min":1, "max":100},),
                "max_num_tokens": ("INT",{"default":49152,"min":0,"max":999999}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, image, seed, pipeline_type, sparse_structure_steps, shape_steps, texture_steps, max_num_tokens):
        image = tensor2pil(image)
        
        sparse_structure_sampler_params = {"steps":sparse_structure_steps}
        shape_slat_sampler_params = {"steps":shape_steps}
        tex_slat_sampler_params = {"steps":texture_steps}
        
        mesh = pipeline.run(image=image, seed=seed, pipeline_type=pipeline_type, sparse_structure_sampler_params = sparse_structure_sampler_params, shape_slat_sampler_params = shape_slat_sampler_params, tex_slat_sampler_params = tex_slat_sampler_params, max_num_tokens = max_num_tokens)[0]
        
        return (mesh,)    

class Trellis2LoadImageWithTransparency:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "Trellis2Wrapper"

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", )
    RETURN_NAMES = ("image", "mask", "image_with_alpha")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        output_images_ori = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            
            output_images_ori.append(pil2tensor(i))

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
            output_image_ori = torch.cat(output_images_ori, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
            output_image_ori = output_images_ori[0]

        return (output_image, output_mask, output_image_ori)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True  

class Trellis2SimplifyMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "target_face_num": ("INT",{"default":1000000,"min":1,"max":16000000}),
                "method": (["Cumesh","Meshlib"],{"default":"Meshlib"}),
            },
        }

    RETURN_TYPES = ("MESHWITHVOXEL", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh, target_face_num, method):        
        if method=="Cumesh":
            mesh.simplify_with_cumesh(target = target_face_num)
        elif method=="Meshlib":
            mesh.simplify_with_meshlib(target = target_face_num)
        else:
            raise Exception("Unknown simplification method")
        
        return (mesh,)          
        
class Trellis2MeshWithVoxelToTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, mesh):        
        trimesh = Trimesh.Trimesh(
            vertices=mesh.vertices.cpu().numpy(),
            faces=mesh.faces.cpu().numpy(),
            process=False
        )
        
        return (trimesh,)
        
class Trellis2ExportMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/Hy3D"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "save_file": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, filename_prefix, file_format, save_file=True):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(exist_ok=True)
        if save_file:
            trimesh.export(output_glb_path, file_type=file_format)
            relative_path = Path(subfolder) / f'{filename}_{counter:05}_.{file_format}'
        else:
            temp_file = Path(full_output_folder, f'hy3dtemp_.{file_format}')
            trimesh.export(temp_file, file_type=file_format)
            relative_path = Path(subfolder) / f'hy3dtemp_.{file_format}'
        
        return (str(relative_path), )        


NODE_CLASS_MAPPINGS = {
    "Trellis2LoadModel": Trellis2LoadModel,
    "Trellis2MeshWithVoxelGenerator": Trellis2MeshWithVoxelGenerator,
    "Trellis2LoadImageWithTransparency": Trellis2LoadImageWithTransparency,
    "Trellis2SimplifyMesh": Trellis2SimplifyMesh,
    "Trellis2MeshWithVoxelToTrimesh": Trellis2MeshWithVoxelToTrimesh,
    "Trellis2ExportMesh": Trellis2ExportMesh,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2LoadModel": "Trellis2 - LoadModel",
    "Trellis2MeshWithVoxelGenerator": "Trellis2 - Mesh With Voxel Generator",
    "Trellis2LoadImageWithTransparency": "Trellis2 - Load Image with Transparency",
    "Trellis2SimplifyMesh": "Trellis2 - Simplify Mesh",
    "Trellis2MeshWithVoxelToTrimesh": "Trellis2 - Mesh With Voxel To Trimesh",
    "Trellis2ExportMesh": "Trellis2 - Export Mesh",
    }
