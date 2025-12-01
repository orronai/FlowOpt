import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import piq
import torch
import torchvision
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from lpips import LPIPS
from PIL import Image

@torch.no_grad()
def calc_v_sd3(
    pipe: StableDiffusion3Pipeline, latent_model_input: torch.Tensor,
    prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor,
    guidance_scale: float, t: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the velocity (v) for Stable Diffusion 3.

    Args:
        pipe (StableDiffusion3Pipeline): The Stable Diffusion 3 pipeline.
        latent_model_input (torch.Tensor): The input latent tensor.
        prompt_embeds (torch.Tensor): The text embeddings for the prompt.
        pooled_prompt_embeds (torch.Tensor): The pooled text embeddings for the prompt.
        guidance_scale (float): The guidance scale for classifier-free guidance.
        t (torch.Tensor): The current timestep.
    Returns:
        torch.Tensor: The predicted noise (velocity).
    """
    timestep = t.expand(latent_model_input.shape[0])

    noise_pred = pipe.transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]

    # perform guidance source
    if pipe.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return noise_pred

# https://github.com/DSL-Lab/UniEdit-Flow
@torch.no_grad()
def uniinv(
    pipe: StableDiffusion3Pipeline, timesteps: torch.Tensor, n_start: int,
    x0_src: torch.Tensor, src_prompt_embeds_all: torch.Tensor,
    src_pooled_prompt_embeds_all: torch.Tensor, src_guidance_scale: float,
) -> torch.Tensor:
    """
    Perform the UniInv inversion process for Stable Diffusion 3.

    Args:
        pipe (StableDiffusion3Pipeline): The Stable Diffusion 3 pipeline.
        timesteps (torch.Tensor): The timesteps for the diffusion process.
        n_start (int): The number of initial timesteps to skip.
        x0_src (torch.Tensor): The source latent tensor.
        src_prompt_embeds_all (torch.Tensor): The text embeddings for the source prompt.
        src_pooled_prompt_embeds_all (torch.Tensor): The pooled text embeddings for the source prompt.
        src_guidance_scale (float): The guidance scale for classifier-free guidance.
    Returns:
        torch.Tensor: The inverted latent tensor.
    """
    x_t = x0_src.clone()
    timesteps_inv = torch.cat([torch.tensor([0.0], device=pipe.device), timesteps.flip(dims=(0,))], dim=0)
    if n_start > 0:
        zipped_timesteps_inv = zip(timesteps_inv[:-n_start - 1], timesteps_inv[1:-n_start])
    else:
        zipped_timesteps_inv = zip(timesteps_inv[:-1], timesteps_inv[1:])
    next_v = None
    for _i, (t_cur, t_prev) in enumerate(zipped_timesteps_inv):
        t_i = t_cur / 1000
        t_ip1 = t_prev / 1000
        dt = t_ip1 - t_i

        if next_v is None:
            latent_model_input = torch.cat([x_t, x_t]) if pipe.do_classifier_free_guidance else (x_t)
            v_tar = calc_v_sd3(
                pipe, latent_model_input, src_prompt_embeds_all,
                src_pooled_prompt_embeds_all, src_guidance_scale, t_cur,
            )
        else:
            v_tar = next_v

        x_t = x_t.to(torch.float32)
        x_t_next = x_t + v_tar * dt
        x_t_next = x_t_next.to(pipe.dtype)

        latent_model_input = torch.cat([x_t_next, x_t_next]) if pipe.do_classifier_free_guidance else (x_t_next)
        v_tar_next = calc_v_sd3(
            pipe, latent_model_input, src_prompt_embeds_all,
            src_pooled_prompt_embeds_all, src_guidance_scale, t_prev,
        )
        next_v = v_tar_next
        x_t = x_t + v_tar_next * dt
        x_t = x_t.to(pipe.dtype)

    return x_t

@torch.no_grad()
def initialization(
    pipe: StableDiffusion3Pipeline, scheduler: FlowMatchEulerDiscreteScheduler,
    T_steps: int, n_start: int, x0_src: torch.Tensor,
    src_prompt: str, negative_prompt: str, src_guidance_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize the inversion process by preparing the latent tensor and prompt embeddings, and performing UniInv.

    Args:
        pipe (StableDiffusion3Pipeline): The Stable Diffusion 3 pipeline.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for the diffusion process.
        T_steps (int): The total number of timesteps for the diffusion process.
        n_start (int): The number of initial timesteps to skip.
        x0_src (torch.Tensor): The source latent tensor.
        src_prompt (str): The source text prompt.
        negative_prompt (str): The negative text prompt for classifier-free guidance.
        src_guidance_scale (float): The guidance scale for classifier-free guidance.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - The inverted latent tensor.
            - The original source latent tensor.
            - The timesteps for the diffusion process.
            - The text embeddings for the source prompt.
            - The pooled text embeddings for the source prompt.
    """
    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=pipe.device,
    )
    src_prompt_embeds_all = torch.cat([src_negative_prompt_embeds, src_prompt_embeds], dim=0) if pipe.do_classifier_free_guidance else src_prompt_embeds
    src_pooled_prompt_embeds_all = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds], dim=0) if pipe.do_classifier_free_guidance else src_pooled_prompt_embeds

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, x0_src.device, timesteps=None)
    pipe._num_timesteps = len(timesteps)

    x_t = uniinv(
        pipe, timesteps, n_start, x0_src, src_prompt_embeds_all,
        src_pooled_prompt_embeds_all, src_guidance_scale,
    )

    return (
        x_t, x0_src, timesteps, src_prompt_embeds_all, src_pooled_prompt_embeds_all,
    )

@torch.no_grad()
def sd3_denoise(
    pipe: StableDiffusion3Pipeline, timesteps: torch.Tensor, n_start: int,
    x_t: torch.Tensor, prompt_embeds_all: torch.Tensor,
    pooled_prompt_embeds_all: torch.Tensor, guidance_scale: float,
) -> torch.Tensor:
    """
    Perform the denoising process for Stable Diffusion 3.

    Args:
        pipe (StableDiffusion3Pipeline): The Stable Diffusion 3 pipeline.
        timesteps (torch.Tensor): The timesteps for the diffusion process.
        n_start (int): The number of initial timesteps to skip.
        x_t (torch.Tensor): The latent tensor at the starting timestep.
        prompt_embeds_all (torch.Tensor): The text embeddings for the prompt.
        pooled_prompt_embeds_all (torch.Tensor): The pooled text embeddings for the prompt.
        guidance_scale (float): The guidance scale for classifier-free guidance.
    Returns:
        torch.Tensor: The denoised latent tensor.
    """
    f_xt = x_t.clone()
    for i, t in enumerate(timesteps[n_start:]):
        t_i = t / 1000
        if i + 1 < len(timesteps[n_start:]):
            t_im1 = (timesteps[n_start + i + 1]) / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        dt = t_im1 - t_i

        latent_model_input = torch.cat([f_xt, f_xt]) if pipe.do_classifier_free_guidance else (f_xt)
        v_tar = calc_v_sd3(
            pipe, latent_model_input, prompt_embeds_all,
            pooled_prompt_embeds_all, guidance_scale, t,
        )
        f_xt = f_xt.to(torch.float32)
        f_xt = f_xt + v_tar * dt
        f_xt = f_xt.to(pipe.dtype)

    return f_xt

@torch.no_grad()
def sd3_inversion(
    pipe: StableDiffusion3Pipeline, scheduler: FlowMatchEulerDiscreteScheduler,
    T_steps: int, n_max: int, x0_src: torch.Tensor, src_prompt: str,
    negative_prompt: str, src_guidance_scale: float, flowopt_iterations: int,
    eta: float, x_0: torch.Tensor, lpips_loss_fn: LPIPS, exp_name: str,
    src_prompt_txt: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Perform the inversion process for Stable Diffusion 3 using FlowOpt.

    Args:
        pipe (StableDiffusion3Pipeline): The Stable Diffusion 3 pipeline.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for the diffusion process.
        T_steps (int): The total number of timesteps for the diffusion process.
        n_max (int): The maximum number of timesteps to consider.
        x0_src (torch.Tensor): The source latent tensor.
        src_prompt (str): The source text prompt.
        negative_prompt (str): The negative text prompt for classifier-free guidance.
        src_guidance_scale (float): The guidance scale for classifier-free guidance.
        flowopt_iterations (int): The number of FlowOpt iterations to perform.
        eta (float): The step size for the FlowOpt update.
        x_0 (torch.Tensor): The original source image tensor.
        lpips_loss_fn (LPIPS): The LPIPS loss function for perceptual similarity.
        exp_name (str): The name of the experiment for saving results.
        src_prompt_txt (str): A text identifier for the source prompt for saving results.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
            - The MSE array over FlowOpt iterations.
            - The LPIPS array over FlowOpt iterations.
            - The SSIM array over FlowOpt iterations.
            - The SSIM of the encoded-decoded image.
            - The LPIPS of the encoded-decoded image.
            - The MSE of the encoded-decoded image.
            - The PSNR of the encoded-decoded image.
    """
    n_start = T_steps - n_max
    (
        x_t, x0_src, timesteps, src_prompt_embeds_all, src_pooled_prompt_embeds_all,
    ) = initialization(
        pipe, scheduler, T_steps, n_start, x0_src, src_prompt, negative_prompt, src_guidance_scale,
    )

    mse_array = np.zeros(flowopt_iterations + 1)
    lpips_array = np.zeros(flowopt_iterations + 1)
    ssim_array = np.zeros(flowopt_iterations + 1)

    j_star = x0_src.clone().to(torch.float32)  # y
    for flowopt_iter in range(flowopt_iterations + 1):
        f_xt = sd3_denoise(
            pipe, timesteps, n_start, x_t, src_prompt_embeds_all,
            src_pooled_prompt_embeds_all, src_guidance_scale,
        )  # Eq. (3)

        f_xt_denorm = (f_xt / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            f_xt_image = pipe.vae.decode(f_xt_denorm, return_dict=False)[0].clamp(-1, 1)
        f_xt_image = pipe.image_processor.postprocess(f_xt_image, output_type="pt")

        if flowopt_iter < flowopt_iterations:
            x_t = x_t.to(torch.float32)
            x_t = x_t - eta * (f_xt - j_star)  # Eq. (6) with c = c_src
            x_t = x_t.to(x0_src.dtype)

        if flowopt_iter == flowopt_iterations:
            x0_src_denorm = (x0_src / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            with torch.autocast("cuda"), torch.inference_mode():
                x_0_enc_dec_image = pipe.vae.decode(x0_src_denorm, return_dict=False)[0].clamp(-1, 1)
            x_0_enc_dec_image = pipe.image_processor.postprocess(x_0_enc_dec_image, output_type="pt")

            enc_dec_img_ssim = piq.ssim(x_0_enc_dec_image.detach().to(torch.float32), x_0.detach().to(torch.float32), data_range=1.0, reduction='sum').item()
            enc_dec_img_lpips = lpips_loss_fn(x_0_enc_dec_image.detach().to(dtype=torch.float32), x_0.detach().to(dtype=torch.float32)).item()
            enc_dec_img_mse = torch.mean((255 * x_0_enc_dec_image[0].detach().cpu().to(torch.float32) - 255 * x_0.detach().cpu().to(torch.float32))**2).numpy()
            enc_dec_img_psnr = 20 * np.log10(255 / np.sqrt(enc_dec_img_mse))

            flowopt_save_dir = os.path.join("saved_info", f"{exp_name}", "SD3", f"eta={eta}", f"src_{src_prompt_txt}")
            os.makedirs(flowopt_save_dir, exist_ok=True)

            torchvision.utils.save_image(
                f_xt_image,
                os.path.join(
                    flowopt_save_dir,
                    f"output_flowopt_iterations={flowopt_iter}_T_steps={T_steps}_n_max={n_max}_cfg={src_guidance_scale}.png",
                ),
            )

        mse_array[flowopt_iter] = torch.mean((255 * f_xt_image[0].detach().cpu().to(torch.float32) - 255 * x_0.detach().cpu().to(torch.float32))**2).numpy()
        lpips_array[flowopt_iter] = lpips_loss_fn(f_xt_image.detach().to(dtype=torch.float32), x_0.detach().to(dtype=torch.float32)).item()
        ssim_array[flowopt_iter] = piq.ssim(f_xt_image.detach().to(torch.float32), x_0.detach().to(torch.float32), data_range=1.0, reduction='sum').item()

    return (
        mse_array, lpips_array, ssim_array,
        enc_dec_img_ssim, enc_dec_img_lpips,
        enc_dec_img_mse, enc_dec_img_psnr,
    )

@torch.no_grad()
def sd3_editing(
    pipe: StableDiffusion3Pipeline, scheduler: FlowMatchEulerDiscreteScheduler,
    T_steps: int, n_max: int, x0_src: torch.Tensor, src_prompt: str,
    tar_prompt: str, negative_prompt: str, src_guidance_scale: float,
    tar_guidance_scale: float, flowopt_iterations: int, eta: float,
    exp_name: str = "", src_prompt_txt: str = "", tar_prompt_txt: str = "",
    gradio_app: bool = False,
) -> Optional[Iterator[List[Tuple[Image.Image, str]]]]:
    """
    Perform the editing process for Stable Diffusion 3 using FlowOpt.

    Args:
        pipe (StableDiffusion3Pipeline): The Stable Diffusion 3 pipeline.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for the diffusion process.
        T_steps (int): The total number of timesteps for the diffusion process.
        n_max (int): The maximum number of timesteps to consider.
        x0_src (torch.Tensor): The source latent tensor.
        src_prompt (str): The source text prompt.
        tar_prompt (str): The target text prompt for editing.
        negative_prompt (str): The negative text prompt for classifier-free guidance.
        src_guidance_scale (float): The guidance scale for the source prompt.
        tar_guidance_scale (float): The guidance scale for the target prompt.
        flowopt_iterations (int): The number of FlowOpt iterations to perform.
        eta (float): The step size for the FlowOpt update.
        exp_name (str): The name of the experiment for saving results.
        src_prompt_txt (str): A text identifier for the source prompt for saving results.
        tar_prompt_txt (str): A text identifier for the target prompt for saving results.
        gradio_app (bool): Whether to yield intermediate results for a gradio app.
    Yields:
        Optional[Iterator[List[Tuple[Image.Image, str]]]]: A list of tuples containing the generated images and their corresponding iteration labels for gradio app.
    """
    n_start = T_steps - n_max
    x_t, x0_src, timesteps, _, _, = initialization(
        pipe, scheduler, T_steps, n_start, x0_src, src_prompt,
        negative_prompt, src_guidance_scale,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=pipe.device,
    )

    tar_prompt_embeds_all = torch.cat([tar_negative_prompt_embeds, tar_prompt_embeds], dim=0) if pipe.do_classifier_free_guidance else tar_prompt_embeds
    tar_pooled_prompt_embeds_all = torch.cat([tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0) if pipe.do_classifier_free_guidance else tar_pooled_prompt_embeds

    if gradio_app:
        history = []
    j_star = x0_src.clone().to(torch.float32)  # y
    for flowopt_iter in range(flowopt_iterations + 1):
        f_xt = sd3_denoise(
            pipe, timesteps, n_start, x_t, tar_prompt_embeds_all,
            tar_pooled_prompt_embeds_all, tar_guidance_scale,
        )  # Eq. (3)

        if flowopt_iter < flowopt_iterations:
            x_t = x_t.to(torch.float32)
            x_t = x_t - eta * (f_xt - j_star)  # Eq. (6) with c = c_tar
            x_t = x_t.to(x0_src.dtype)

        x0_flowopt = f_xt.clone()
        x0_flowopt_denorm = (x0_flowopt / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            x0_flowopt_image = pipe.vae.decode(x0_flowopt_denorm, return_dict=False)[0].clamp(-1, 1)
        if gradio_app:
            x0_flowopt_image_pil = pipe.image_processor.postprocess(x0_flowopt_image)[0]
            history.append((x0_flowopt_image_pil, f"Iteration {flowopt_iter}"))
            yield history
        else:
            x0_flowopt_image = pipe.image_processor.postprocess(x0_flowopt_image, output_type="pt")

            flowopt_save_dir = os.path.join("saved_info", f"{exp_name}", "SD3", f"eta={eta}", f"src_{src_prompt_txt}", f"tar_{tar_prompt_txt}")
            os.makedirs(flowopt_save_dir, exist_ok=True)

            torchvision.utils.save_image(
                x0_flowopt_image,
                os.path.join(
                    flowopt_save_dir,
                    f"output_flowopt_iterations={flowopt_iter}_T_steps={T_steps}_n_max={n_max}_cfg={tar_guidance_scale}.png",
                ),
            )

            if flowopt_iter == flowopt_iterations:
                with open(f"{flowopt_save_dir}/prompts.txt", "w") as f:
                    f.write(f"Source prompt: {src_prompt}\n")
                    f.write(f"Target prompt: {tar_prompt}\n")
