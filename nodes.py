# ComfyUI custom node for HunyuanVideo-Foley with explicit model dropdowns
# Place this file at: ComfyUI/custom_nodes/hunyuanvideo_foley_nodes/nodes.py

import os
import traceback
from typing import List, Tuple
import torchaudio
import torch
from loguru import logger

# ComfyUI helpers
try:
    from folder_paths import models_dir
except Exception:
    # Minimal fallback (should not happen in normal ComfyUI installs)
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")

# Project utils (identical imports kept; only path-handling added in utils/model_utils.py)
from .hunyuanvideo_foley.utils.model_utils import load_all_models  # new helper we add below in model_utils.py
from .hunyuanvideo_foley.utils.feature_utils import feature_process
from .hunyuanvideo_foley.utils.media_utils import merge_audio_video
from .hunyuanvideo_foley.utils.model_utils import denoise_process


NODE_CATEGORY = "Audio/HunyuanVideo-Foley"
NODE_NAME = "Hunyuan Foley: Generate Audio"

def _list_files_safe(folder: str, exts: Tuple[str, ...] = None) -> List[str]:
    if not os.path.isdir(folder):
        return []
    exts = exts or (".pth", ".pt", ".safetensors", ".bin", ".ckpt")
    items = []
    for root, dirs, files in os.walk(folder):
        rel_path = os.path.relpath(root, folder)
        # Nur Modell-Dateien mit relativem Pfad
        for f in files:
            if any(f.lower().endswith(e) for e in exts):
                if rel_path == ".":
                    items.append(f)
                else:
                    items.append(os.path.join(rel_path, f).replace("\\", "/"))
    return sorted(items)

def _list_configs(config_dir: str) -> List[str]:
    return [n for n in sorted(os.listdir(config_dir)) if n.lower().endswith((".yml", ".yaml"))]

def _default_models_root() -> str:
    return models_dir if os.path.isdir(models_dir) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def _get_next_filename(base_dir: str, base_name: str, extension: str) -> str:
    """Find next available numbered filename"""
    os.makedirs(base_dir, exist_ok=True)
    counter = 1
    while True:
        filename = f"{base_name}_{counter:05d}{extension}"
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            return filepath
        counter += 1

class HunyuanFoleyNode:
    """
    UI:
      - video_path: Pfad zu einem Video
      - prompt: Textprompt
      - config_name: Auswahl aus custom_nodes/<this>/config/
      - diffusion_model / vae_model / clip_vision_model / clap_model: Auswahl aus ComfyUI/models/<subdir>
      - guidance_scale, steps, save_video

    Output:
      - audio_file (WAV Pfad)
      - optional merged_video (MP4 Pfad)
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Model folders
        mroot = _default_models_root()
        diffusion_dir = os.path.join(mroot, "diffusion_models")
        vae_dir       = os.path.join(mroot, "vae")
        clipv_dir     = os.path.join(mroot, "clip_vision")
        clap_dir      = os.path.join(mroot, "clap")  # as requested

        # enumerate files (common weight extensions)
        weight_exts = (".pth", ".pt", ".safetensors", ".bin")

        diffusion_choices = _list_files_safe(diffusion_dir, weight_exts)
        vae_choices       = _list_files_safe(vae_dir, weight_exts)
        clipv_choices     = _list_files_safe(clipv_dir, weight_exts) or _list_files_safe(clipv_dir)  # also allow directories (transformers)
        clap_choices      = _list_files_safe(clap_dir, weight_exts) or _list_files_safe(clap_dir)    # also allow directories

        # configs from this node's config folder
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(this_dir, "config")
        config_choices = _list_configs(config_dir) if os.path.isdir(config_dir) else []

        if not config_choices:
            config_choices = ["<put_your_yaml_in_config_folder>"]

        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "config_name": (config_choices, {"default": config_choices[0]}),

                "diffusion_model": (diffusion_choices or ["<no files in models/diffusion_models>"],),
                "vae_model": (vae_choices or ["<no files in models/vae>"],),
                "clip_vision_model": (clipv_choices or ["<no files in models/clip_vision>"],),
                "clap_model": (clap_choices or ["<no files in models/clap>"],),

                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "save_video": ("BOOLEAN", {"default": True}),
                "save_audio": ("BOOLEAN", {"default": True}),

            },
            "optional": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "video": ("IMAGE",),  # Frame batch von Video Loader
                "video_fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": 15}),
                "output_dir": ("STRING", {"default": "outputs/hunyuan_foley"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING", "STRING")
    RETURN_NAMES = ("audio", "sample_rate", "audio_wav_path", "merged_video_path")
    FUNCTION = "run"
    CATEGORY = NODE_CATEGORY

    def _select_device(self, device_str: str, gpu_id: int):
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device(f"cuda:{gpu_id}")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if device_str == "cuda":
            return torch.device(f"cuda:{gpu_id}")
        return torch.device(device_str)

    def run(
        self,
        prompt: str,
        config_name: str,
        diffusion_model: str,
        vae_model: str,
        clip_vision_model: str,
        clap_model: str,
        guidance_scale: float,
        num_inference_steps: int,
        save_video: bool,
        save_audio: bool,
        video_path: str = "",
        video = None,  # Frame batch
        video_fps: float = 16.0,
        device: str = "auto",
        gpu_id: int = 0,
        output_dir: str = "output/hunyuan_foley",
    ):
        try:
                    # Video-Handling
            if video is not None:
                # Frames zu temporärem Video speichern
                import tempfile
                import cv2
                import numpy as np
                
                # ComfyUI Format: [frames, height, width, channels] mit Werten 0-1
                if isinstance(video, torch.Tensor):
                    frames_np = video.cpu().numpy()
                else:
                    frames_np = video
            
                # Debug info
                logger.info(f"[Hunyuan Foley] Received frames shape: {frames_np.shape}")
                logger.info(f"[Hunyuan Foley] Frames min/max: {frames_np.min()}/{frames_np.max()}")
    
                temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                temp_path = temp_video.name
                temp_video.close()
            
                # Video Writer setup
                num_frames, h, w, c = frames_np.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_path, fourcc, video_fps, (w, h))
            
                # Frames konvertieren und schreiben
                for i in range(num_frames):
                    frame = frames_np[i]
                
                    # Skalierung auf 0-255 falls nötig
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                    # RGB zu BGR für OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
            
                out.release()
                video_path = temp_path
                logger.info(f"[Hunyuan Foley] Saved temporary video to: {temp_path}")
            
            elif not video_path:
                raise ValueError("Either video_path or video frames must be provided")
        
            # Überprüfen ob Video existiert und lesbar ist
            if not os.path.exists(video_path):
                raise ValueError(f"Video file not found: {video_path}")
                
            # Resolve all paths
            node_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(node_dir, "config", config_name)

            mroot = _default_models_root()
            diffusion_path = os.path.join(mroot, "diffusion_models", diffusion_model)
            vae_path       = os.path.join(mroot, "vae", vae_model)
            clipv_path     = os.path.join(mroot, "clip_vision", clip_vision_model)
            clap_path      = os.path.join(mroot, "clap", clap_model)

            # If user selected a file in clip_vision/clap which actually belongs to a folder-based Transformers model,
            # normalize: if it's a file, use its parent directory for AutoModel.from_pretrained
            def normalize_dir_for_transformers(p: str) -> str:
                if os.path.isdir(p):
                    return p
                parent = os.path.dirname(p)
                return parent if os.path.isdir(parent) else p  # last resort: let utils handle

            clipv_path = normalize_dir_for_transformers(clipv_path)
            clap_path  = normalize_dir_for_transformers(clap_path)

            # Synchformer default location: ComfyUI/syncforner/synchformer_state_dict.pth
            comfy_root = os.path.dirname(os.path.dirname(node_dir))  # go up to ComfyUI/
            syncformer_root = os.path.join(comfy_root, "syncforner")
            syncformer_ckpt = os.path.join(syncformer_root, "synchformer_state_dict.pth")

            # Device
            torch_device = self._select_device(device, gpu_id)
            logger.info(f"[Hunyuan Foley Node] Using device: {torch_device}")

            # Load all models via our updated utils (keeps original import style, only paths added)
            model_dict, cfg = load_all_models(
                diffusion_model_path=diffusion_path,
                vae_path=vae_path,
                clip_vision_path=clipv_path,
                clap_path=clap_path,
                config_path=config_path,
                syncformer_ckpt=syncformer_ckpt,
                device=torch_device,
            )

            # Prepare output names
            base_name = os.path.splitext(os.path.basename(output_dir))[0] if output_dir else "hunyuan_foley"
            output_folder = os.path.dirname(output_dir) if os.path.dirname(output_dir) else "output"

            # Feature + denoise (uses your existing pipeline)
            visual_feats, text_feats, audio_len_in_s = feature_process(
                video_path, prompt, model_dict, cfg
            )

            # Override audio length if frames provided
            if video is not None:
                audio_len_in_s = num_frames / video_fps
                logger.info(f"[Hunyuan Foley] Audio length set to {audio_len_in_s}s")

            audio_tensor, sr = denoise_process(
                visual_feats,
                text_feats,
                audio_len_in_s,
                model_dict,
                cfg,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                batch_size=1,
            )

            # Save audio if requested
            audio_out = ""
            if save_audio:
                audio_out = _get_next_filename(output_folder, base_name, ".wav")
                torchaudio.save(audio_out, audio_tensor[0], sr)
                logger.info(f"[Hunyuan Foley] Saved audio to: {audio_out}")

            # Save merged video if requested
            # Save merged video if requested
            merged_out = ""
            if save_video and os.path.exists(video_path):
                if save_audio and audio_out:
                    # Audio wurde bereits gespeichert, nutze es
                    merged_out = _get_next_filename(output_folder, base_name, ".mp4")
                    merge_audio_video(audio_out, video_path, merged_out)
                else:
                    # Audio nicht gespeichert, aber Video mit Audio gewünscht
                    import tempfile
                    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    torchaudio.save(temp_audio.name, audio_tensor[0], sr)
                    merged_out = _get_next_filename(output_folder, base_name, ".mp4")
                    merge_audio_video(temp_audio.name, video_path, merged_out)
                    os.unlink(temp_audio.name)  # Temp-Audio löschen
                logger.info(f"[Hunyuan Foley] Saved merged video to: {merged_out}")

            # Audio tensor for further processing
            audio_output = {
                "waveform": audio_tensor.cpu(),
                "sample_rate": sr
            }

            return (audio_output, sr, audio_out, merged_out)
        
        except Exception as e:
            logger.error(f"[Hunyuan Foley Node] Error: {e}")
            # Dummy-Outputs bei Fehler
            empty_audio = {"waveform": torch.zeros((1, 1, 48000)), "sample_rate": 48000}
            return (empty_audio, 48000, "", "")


# ---- ComfyUI node registration ----
NODE_CLASS_MAPPINGS = {
    "HunyuanFoleyNode": HunyuanFoleyNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanFoleyNode": NODE_NAME,
}
