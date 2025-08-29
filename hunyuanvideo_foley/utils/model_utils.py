# utils/model_utils.py
# Replacement for path-flexible loading. Based on your uploaded version; imports kept.

import os
import torch
from loguru import logger
from torchvision import transforms
from torchvision.transforms import v2
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, AutoModel, ClapTextModelWithProjection

# original local imports stay as-is (paths relative to your package)
from ..models.dac_vae.model.dac import DAC
from ..models.synchformer import Synchformer
from ..models.hifi_foley import HunyuanVideoFoley
from .config_utils import load_yaml, AttributeDict
from .schedulers import FlowMatchDiscreteScheduler
from tqdm import tqdm


def load_state_dict(model, model_path):
    logger.info(f"Loading model state dict from: {model_path}")
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys in state dict ({len(missing_keys)} keys):")
        for key in missing_keys:
            logger.warning(f"  - {key}")
    else:
        logger.info("No missing keys found")

    if unexpected_keys:
        logger.warning(f"Unexpected keys in state dict ({len(unexpected_keys)} keys):")
        for key in unexpected_keys:
            logger.warning(f"  - {key}")
    else:
        logger.info("No unexpected keys found")

    logger.info("Model state dict loaded successfully")
    return model


def _as_transformers_dir(p: str) -> str:
    """If a file is given, return its parent folder for transformers; else return p."""
    if os.path.isdir(p):
        return p
    parent = os.path.dirname(p)
    return parent if os.path.isdir(parent) else p


def load_all_models(
    diffusion_model_path: str,
    vae_path: str,
    clip_vision_path: str,
    clap_path: str,
    config_path: str,
    syncformer_ckpt: str,
    device: torch.device,
):
    """
    Load all models from explicit paths:
      - diffusion_model_path: e.g. ComfyUI/models/diffusion_models/hunyuanvideo_foley.pth
      - vae_path:             e.g. ComfyUI/models/vae/foley_vae_128d_48k.pth
      - clip_vision_path:     e.g. ComfyUI/models/clip_vision/<folder or file>
      - clap_path:            e.g. ComfyUI/models/clap/<folder or file>
      - config_path:          e.g. custom_nodes/<this>/config/<cfg>.yaml
      - syncformer_ckpt:      e.g. ComfyUI/syncforner/synchformer_state_dict.pth
    """
    logger.info("Starting model loading process...")
    logger.info(f"Config: {config_path}")
    logger.info(f"Diffusion weights: {diffusion_model_path}")
    logger.info(f"VAE weights: {vae_path}")
    logger.info(f"CLIP-Vision path: {clip_vision_path}")
    logger.info(f"CLAP path: {clap_path}")
    logger.info(f"Synchformer ckpt: {syncformer_ckpt}")
    logger.info(f"Target device: {device}")

    cfg = load_yaml(config_path)
    logger.info("Configuration loaded successfully")

    # HunyuanVideoFoley (diffusion / main)
    logger.info("Loading HunyuanVideoFoley main model…")
    foley_model = HunyuanVideoFoley(cfg, dtype=torch.bfloat16, device=device).to(device=device, dtype=torch.bfloat16)
    foley_model = load_state_dict(foley_model, diffusion_model_path)
    foley_model.eval()
    logger.info("HunyuanVideoFoley model loaded and set to eval")

    # DAC-VAE
    logger.info(f"Loading DAC VAE from: {vae_path}")
    dac_model = DAC.load(vae_path)
    dac_model = dac_model.to(device)
    dac_model.requires_grad_(False)
    dac_model.eval()
    logger.info("DAC VAE loaded")

    # SigLIP2 visual-encoder
    logger.info("Loading SigLIP2 visual encoder…")
    siglip2_preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # Accept folder or file; prefer folder for transformers
    clipv_dir = _as_transformers_dir(clip_vision_path)
    try:
        siglip2_model = AutoModel.from_pretrained(clipv_dir).to(device).eval()
    except Exception as e:
        logger.warning(f"AutoModel.from_pretrained failed for '{clipv_dir}': {e}")
        # last-resort: try torch.load then .to(device) if it returns a nn.Module
        maybe = torch.load(clip_vision_path, map_location="cpu")
        if hasattr(maybe, "to"):
            siglip2_model = maybe.to(device).eval()
        else:
            raise RuntimeError(f"Cannot load CLIP-Vision model from '{clip_vision_path}' or '{clipv_dir}'")

    logger.info("SigLIP2 model and preprocessing loaded")

    # CLAP text-encoder
    logger.info("Loading CLAP text encoder…")
    clap_dir = _as_transformers_dir(clap_path)
    try:
        clap_tokenizer = AutoTokenizer.from_pretrained(clap_dir)
        clap_model = ClapTextModelWithProjection.from_pretrained(clap_dir).to(device)
    except Exception as e:
        logger.warning(f"CLAP from_pretrained failed for '{clap_dir}': {e}")
        # As a fallback, try to load only the model
        clap_tokenizer = None
        maybe = torch.load(clap_path, map_location="cpu")
        if hasattr(maybe, "to"):
            clap_model = maybe.to(device).eval()
        else:
            raise RuntimeError(f"Cannot load CLAP from '{clap_path}' or '{clap_dir}'")
    logger.info("CLAP tokenizer/model loaded")

    # Synchformer
    logger.info(f"Loading Synchformer from: {syncformer_ckpt}")
    syncformer_preprocess = v2.Compose(
        [
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    syncformer_model = Synchformer()
    syncformer_model.load_state_dict(torch.load(syncformer_ckpt, weights_only=False, map_location="cpu"))
    syncformer_model = syncformer_model.to(device).eval()
    logger.info("Synchformer loaded")

    model_dict = AttributeDict(
        {
            "foley_model": foley_model,
            "dac_model": dac_model,
            "siglip2_preprocess": siglip2_preprocess,
            "siglip2_model": siglip2_model,
            "clap_tokenizer": clap_tokenizer,
            "clap_model": clap_model,
            "syncformer_preprocess": syncformer_preprocess,
            "syncformer_model": syncformer_model,
            "device": device,
        }
    )

    logger.info("All models loaded successfully")
    return model_dict, cfg


# ===== below: unchanged generation utilities from your file =====

def retrieve_timesteps(scheduler, num_inference_steps, device, **kwargs):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def prepare_latents(scheduler, batch_size, num_channels_latents, length, dtype, device):
    shape = (batch_size, num_channels_latents, int(length))
    latents = randn_tensor(shape, device=device, dtype=dtype)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    return latents


@torch.no_grad()
def denoise_process(visual_feats, text_feats, audio_len_in_s, model_dict, cfg, guidance_scale=4.5, num_inference_steps=50, batch_size=1):
    target_dtype = model_dict.foley_model.dtype
    autocast_enabled = target_dtype != torch.float32
    device = model_dict.device

    scheduler = FlowMatchDiscreteScheduler(
        shift=cfg.diffusion_config.sample_flow_shift,
        reverse=cfg.diffusion_config.flow_reverse,
        solver=cfg.diffusion_config.flow_solver,
        use_flux_shift=cfg.diffusion_config.sample_use_flux_shift,
        flux_base_shift=cfg.diffusion_config.flux_base_shift,
        flux_max_shift=cfg.diffusion_config.flux_max_shift,
    )

    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device)
    latents = prepare_latents(
        scheduler,
        batch_size=batch_size,
        num_channels_latents=cfg.model_config.model_kwargs.audio_vae_latent_dim,
        length=audio_len_in_s * cfg.model_config.model_kwargs.audio_frame_rate,
        dtype=target_dtype,
        device=device,
    )

    from tqdm import tqdm as _tqdm
    for i, t in _tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising steps"):
        latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        latent_input = scheduler.scale_model_input(latent_input, t)
        t_expand = t.repeat(latent_input.shape[0])

        # siglip2 features
        siglip2_feat = visual_feats.siglip2_feat.repeat(batch_size, 1, 1)
        uncond_siglip2_feat = model_dict.foley_model.get_empty_clip_sequence(bs=batch_size, len=siglip2_feat.shape[1]).to(device)
        siglip2_feat_input = torch.cat([uncond_siglip2_feat, siglip2_feat], dim=0) if guidance_scale and guidance_scale > 1.0 else siglip2_feat

        # syncformer features
        syncformer_feat = visual_feats.syncformer_feat.repeat(batch_size, 1, 1)
        uncond_syncformer_feat = model_dict.foley_model.get_empty_sync_sequence(bs=batch_size, len=syncformer_feat.shape[1]).to(device)
        syncformer_feat_input = torch.cat([uncond_syncformer_feat, syncformer_feat], dim=0) if guidance_scale and guidance_scale > 1.0 else syncformer_feat

        # text features
        text_feat_repeated = text_feats.text_feat.repeat(batch_size, 1, 1)
        uncond_text_feat_repeated = text_feats.uncond_text_feat.repeat(batch_size, 1, 1)
        text_feat_input = torch.cat([uncond_text_feat_repeated, text_feat_repeated], dim=0) if guidance_scale and guidance_scale > 1.0 else text_feat_repeated

        with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=target_dtype):
            noise_pred = model_dict.foley_model(
                x=latent_input,
                t=t_expand,
                cond=text_feat_input,
                clip_feat=siglip2_feat_input,
                sync_feat=syncformer_feat_input,
                return_dict=True,
            )["x"]

        noise_pred = noise_pred.to(dtype=torch.float32)

        if guidance_scale and guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    with torch.no_grad():
        audio = model_dict.dac_model.decode(latents).float().cpu()

    audio = audio[:, : int(audio_len_in_s * model_dict.dac_model.sample_rate)]
    return audio, model_dict.dac_model.sample_rate
