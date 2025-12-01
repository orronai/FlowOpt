import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import piq
import torch
import torchvision
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from lpips import LPIPS
from PIL import Image

@torch.no_grad()
def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

@torch.no_grad()
def calc_v_flux(
    pipe: FluxPipeline, latents: torch.Tensor, prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor, guidance: torch.Tensor,
    text_ids: torch.Tensor, latent_image_ids: torch.Tensor, t: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the velocity (v) for FLUX.

    Args:
        pipe (FluxPipeline): The FLUX pipeline.
        latents (torch.Tensor): The latent tensor at the current timestep.
        prompt_embeds (torch.Tensor): The prompt embeddings.
        pooled_prompt_embeds (torch.Tensor): The pooled prompt embeddings.
        guidance (torch.Tensor): The guidance scale tensor.
        text_ids (torch.Tensor): The text token IDs.
        latent_image_ids (torch.Tensor): The latent image token IDs.
        t (torch.Tensor): The current timestep.
    Returns:
        torch.Tensor: The predicted noise (velocity).
    """
    timestep = t.expand(latents.shape[0])

    noise_pred = pipe.transformer(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]

    return noise_pred

@torch.no_grad()
def prep_input(
    pipe: FluxPipeline, scheduler: FlowMatchEulerDiscreteScheduler,
    T_steps: int, x0_src: torch.Tensor, src_prompt: str,
    src_guidance_scale: float,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """
    Prepare the input tensors for the FLUX pipeline.

    Args:
        pipe (FluxPipeline): The FLUX pipeline.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for the diffusion process.
        T_steps (int): The total number of timesteps for the diffusion process.
        x0_src (torch.Tensor): The source latent tensor.
        src_prompt (str): The source text prompt.
        src_guidance_scale (float): The guidance scale for classifier-free guidance.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
            - Prepared source latent tensor.
            - Latent image token IDs.
            - Timesteps tensor for the diffusion process.
            - Original height of the input image.
            - Original width of the input image.
            - Source prompt embeddings.
            - Source pooled prompt embeddings.
            - Source text token IDs.
    """
    orig_height, orig_width = x0_src.shape[2] * pipe.vae_scale_factor, x0_src.shape[3] * pipe.vae_scale_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x0_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x0_src.shape[0], num_channels_latents=num_channels_latents,
        height=orig_height, width=orig_width,
        dtype=x0_src.dtype, device=x0_src.device, generator=None, latents=x0_src,
    )
    x0_src = pipe._pack_latents(x0_src, x0_src.shape[0], num_channels_latents, x0_src.shape[2], x0_src.shape[3])

    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x0_src.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        x0_src.device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=x0_src.device,
    )

    return (
        x0_src, latent_src_image_ids, timesteps, orig_height, orig_width,
        src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids
    )

# https://github.com/DSL-Lab/UniEdit-Flow
@torch.no_grad()
def uniinv(
    pipe: FluxPipeline, scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: torch.Tensor, n_start: int, x0_src: torch.Tensor,
    src_prompt_embeds: torch.Tensor, src_pooled_prompt_embeds: torch.Tensor,
    src_guidance: torch.Tensor, src_text_ids: torch.Tensor,
    latent_src_image_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Perform the UniInv inversion process for FLUX.

    Args:
        pipe (FluxPipeline): The FLUX pipeline.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for the diffusion process.
        timesteps (torch.Tensor): The timesteps for the diffusion process.
        n_start (int): The number of initial timesteps to skip.
        x0_src (torch.Tensor): The source latent tensor.
        src_prompt_embeds (torch.Tensor): The source prompt embeddings.
        src_pooled_prompt_embeds (torch.Tensor): The source pooled prompt embeddings.
        src_guidance (torch.Tensor): The guidance scale tensor.
        src_text_ids (torch.Tensor): The source text token IDs.
        latent_src_image_ids (torch.Tensor): The latent image token IDs.
    Returns:
        torch.Tensor: The inverted latent tensor.
    """
    x_t = x0_src.clone()
    timesteps_inv = timesteps.flip(dims=(0,))[:-n_start] if n_start > 0 else timesteps.flip(dims=(0,))
    next_v = None
    for _i, t in enumerate(timesteps_inv):
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        t_ip1 = scheduler.sigmas[scheduler.step_index + 1]
        dt = t_i - t_ip1

        if next_v is None:
            v_tar = calc_v_flux(
                pipe, latents=x_t, prompt_embeds=src_prompt_embeds,
                pooled_prompt_embeds=src_pooled_prompt_embeds, guidance=src_guidance,
                text_ids=src_text_ids, latent_image_ids=latent_src_image_ids, t=t_ip1 * 1000,
            )
        else:
            v_tar = next_v
        x_t = x_t.to(torch.float32)
        x_t_next = x_t + v_tar * dt
        x_t_next = x_t_next.to(pipe.dtype)

        v_tar_next = calc_v_flux(
            pipe, latents=x_t_next, prompt_embeds=src_prompt_embeds,
            pooled_prompt_embeds=src_pooled_prompt_embeds, guidance=src_guidance,
            text_ids=src_text_ids, latent_image_ids=latent_src_image_ids, t=t,
        )
        next_v = v_tar_next
        x_t = x_t + v_tar_next * dt
        x_t = x_t.to(pipe.dtype)

    return x_t

@torch.no_grad()
def initialization(
    pipe: FluxPipeline, scheduler: FlowMatchEulerDiscreteScheduler,
    T_steps: int, n_start: int, x0_src: torch.Tensor, src_prompt: str,
    src_guidance_scale: float,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """
    Initialize the inversion process by preparing the latent tensor and prompt embeddings, and performing UniInv.

    Args:
        pipe (FluxPipeline): The FLUX pipeline.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for the diffusion process.
        T_steps (int): The total number of timesteps for the diffusion process.
        n_start (int): The number of initial timesteps to skip.
        x0_src (torch.Tensor): The source latent tensor.
        src_prompt (str): The source text prompt.
        src_guidance_scale (float): The guidance scale for classifier-free guidance.
    Returns:
        Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        ]:
            - The inverted latent tensor.
            - The source latent tensor.
            - The timesteps for the diffusion process.
            - The latent image token IDs.
            - The original height of the input image.
            - The original width of the input image.
            - The source prompt embeddings.
            - The source pooled prompt embeddings.
            - The source text token IDs.
            - The guidance scale tensor.
    """
    (
        x0_src, latent_src_image_ids, timesteps, orig_height, orig_width,
        src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids
    ) = prep_input(pipe, scheduler, T_steps, x0_src, src_prompt, src_guidance_scale)

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=pipe.device)
        src_guidance = src_guidance.expand(x0_src.shape[0])
    else:
        src_guidance = None

    x_t = uniinv(
        pipe, scheduler, timesteps, n_start, x0_src,
        src_prompt_embeds, src_pooled_prompt_embeds, src_guidance,
        src_text_ids, latent_src_image_ids,
    )

    return (
        x_t, x0_src, timesteps, latent_src_image_ids, orig_height, orig_width,
        src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids, src_guidance,
    )

@torch.no_grad()
def flux_denoise(
    pipe: FluxPipeline, scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: torch.Tensor, n_start: int, x_t: torch.Tensor,
    prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor,
    guidance: torch.Tensor, text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Perform the denoising process for FLUX.

    Args:
        pipe (FluxPipeline): The FLUX pipeline.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for the diffusion process.
        timesteps (torch.Tensor): The timesteps for the diffusion process.
        n_start (int): The number of initial timesteps to skip.
        x_t (torch.Tensor): The latent tensor at the starting timestep.
        prompt_embeds (torch.Tensor): The prompt embeddings.
        pooled_prompt_embeds (torch.Tensor): The pooled prompt embeddings.
        guidance (torch.Tensor): The guidance scale tensor.
        text_ids (torch.Tensor): The text token IDs.
        latent_image_ids (torch.Tensor): The latent image token IDs.
    Returns:
        torch.Tensor: The denoised latent tensor.
    """
    f_xt = x_t.clone()
    for _i, t in enumerate(timesteps[n_start:]):
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        dt = t_im1 - t_i

        v_tar = calc_v_flux(
            pipe, latents=f_xt, prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds, guidance=guidance,
            text_ids=text_ids, latent_image_ids=latent_image_ids, t=t,
        )
        f_xt = f_xt.to(torch.float32)
        f_xt = f_xt + v_tar * dt
        f_xt = f_xt.to(pipe.dtype)

    return f_xt

@torch.no_grad()
def flux_inversion(
    pipe: FluxPipeline, scheduler: FlowMatchEulerDiscreteScheduler,
    T_steps: int, n_max: int, x0_src: torch.Tensor, src_prompt: str,
    src_guidance_scale: float, flowopt_iterations: int, eta: float,
    x_0: torch.Tensor, lpips_loss_fn: LPIPS, exp_name: str,
    src_prompt_txt: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Perform the inversion process for FLUX using FlowOpt.

    Args:
        pipe (FluxPipeline): The FLUX pipeline.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for the diffusion process.
        T_steps (int): The total number of timesteps for the diffusion process.
        n_max (int): The maximum number of timesteps to consider.
        x0_src (torch.Tensor): The source latent tensor.
        src_prompt (str): The source text prompt.
        src_guidance_scale (float): The guidance scale for classifier-free guidance.
        flowopt_iterations (int): The number of FlowOpt iterations to perform.
        eta (float): The step size for the FlowOpt update.
        x_0 (torch.Tensor): The original source image tensor.
        lpips_loss_fn (LPIPS): The LPIPS loss function for perceptual similarity.
        exp_name (str): The name of the experiment for saving results.
        src_prompt_txt (str): A text identifier for the source prompt for saving results.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
            - MSE array over FlowOpt iterations.
            - LPIPS array over FlowOpt iterations.
            - SSIM array over FlowOpt iterations.
            - SSIM of the encoded-decoded image.
            - LPIPS of the encoded-decoded image.
            - MSE of the encoded-decoded image.
            - PSNR of the encoded-decoded image.
    """
    n_start = T_steps - n_max
    (
        x_t, x0_src, timesteps, latent_src_image_ids, orig_height, orig_width,
        src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids, src_guidance,
    ) = initialization(
        pipe, scheduler, T_steps, n_start, x0_src, src_prompt, src_guidance_scale
    )

    mse_array = np.zeros(flowopt_iterations + 1)
    lpips_array = np.zeros(flowopt_iterations + 1)
    ssim_array = np.zeros(flowopt_iterations + 1)

    j_star = x0_src.clone().to(torch.float32)  # y
    for flowopt_iter in range(flowopt_iterations + 1):
        f_xt = flux_denoise(
            pipe, scheduler, timesteps, n_start, x_t,
            src_prompt_embeds, src_pooled_prompt_embeds, src_guidance,
            src_text_ids, latent_src_image_ids,
        )  # Eq. (3)

        unpacked_f_xt = pipe._unpack_latents(f_xt, orig_height, orig_width, pipe.vae_scale_factor)
        f_xt_denorm = (unpacked_f_xt / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            f_xt_image = pipe.vae.decode(f_xt_denorm, return_dict=False)[0].clamp(-1, 1)
        f_xt_image = pipe.image_processor.postprocess(f_xt_image, output_type="pt")

        if flowopt_iter < flowopt_iterations:
            x_t = x_t.to(torch.float32)
            x_t = x_t - eta * (f_xt - j_star)  # Eq. (6) with c = c_src
            x_t = x_t.to(x0_src.dtype)

        if flowopt_iter == flowopt_iterations:
            unpacked_x0_src = pipe._unpack_latents(x0_src, orig_height, orig_width, pipe.vae_scale_factor)
            x0_src_denorm = (unpacked_x0_src / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            with torch.autocast("cuda"), torch.inference_mode():
                x_0_enc_dec_image = pipe.vae.decode(x0_src_denorm, return_dict=False)[0].clamp(-1, 1)
            x_0_enc_dec_image = pipe.image_processor.postprocess(x_0_enc_dec_image, output_type="pt")

            enc_dec_img_ssim = piq.ssim(x_0_enc_dec_image.detach().to(torch.float32), x_0.detach().to(torch.float32), data_range=1.0, reduction='sum').item()
            enc_dec_img_lpips = lpips_loss_fn(x_0_enc_dec_image.detach().to(dtype=torch.float32), x_0.detach().to(dtype=torch.float32)).item()
            enc_dec_img_mse = torch.mean((255 * x_0_enc_dec_image[0].detach().cpu().to(torch.float32) - 255 * x_0.detach().cpu().to(torch.float32))**2).numpy()
            enc_dec_img_psnr = 20 * np.log10(255 / np.sqrt(enc_dec_img_mse))

            flowopt_save_dir = os.path.join("saved_info", f"{exp_name}", "FLUX", f"eta={eta}", f"src_{src_prompt_txt}")
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
def flux_editing(
    pipe: FluxPipeline, scheduler: FlowMatchEulerDiscreteScheduler,
    T_steps: int, n_max: int, x0_src: torch.Tensor, src_prompt: str,
    tar_prompt: str, src_guidance_scale: float, tar_guidance_scale: float,
    flowopt_iterations: int, eta: float, exp_name: str = "",
    src_prompt_txt: str = "", tar_prompt_txt: str = "",
    gradio_app: bool = False,
) -> Optional[Iterator[List[Tuple[Image.Image, str]]]]:
    """
    Perform the editing process for FLUX using FlowOpt.

    Args:
        pipe (FluxPipeline): The FLUX pipeline.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for the diffusion process.
        T_steps (int): The total number of timesteps for the diffusion process.
        n_max (int): The maximum number of timesteps to consider.
        x0_src (torch.Tensor): The source latent tensor.
        src_prompt (str): The source text prompt.
        tar_prompt (str): The target text prompt for editing.
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
    (
        x_t, x0_src, timesteps, latent_src_image_ids, orig_height, orig_width,
        _, _, _, _,
    ) = initialization(
        pipe, scheduler, T_steps, n_start, x0_src, src_prompt, src_guidance_scale,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        pooled_tar_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=pipe.device,
    )

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        tar_guidance = torch.tensor([tar_guidance_scale], device=pipe.device)
        tar_guidance = tar_guidance.expand(x0_src.shape[0])
    else:
        tar_guidance = None

    if gradio_app:
        history = []
    j_star = x0_src.clone().to(torch.float32)  # y
    for flowopt_iter in range(flowopt_iterations + 1):
        f_xt = flux_denoise(
            pipe, scheduler, timesteps, n_start, x_t,
            tar_prompt_embeds, pooled_tar_prompt_embeds, tar_guidance,
            tar_text_ids, latent_src_image_ids,
        )  # Eq. (3)

        if flowopt_iter < flowopt_iterations:
            x_t = x_t.to(torch.float32)
            x_t = x_t - eta * (f_xt - j_star)  # Eq. (6) with c = c_tar
            x_t = x_t.to(x0_src.dtype)

        x0_flowopt = f_xt.clone()
        unpacked_x0_flowopt = pipe._unpack_latents(x0_flowopt, orig_height, orig_width, pipe.vae_scale_factor)
        x0_flowopt_denorm = (unpacked_x0_flowopt / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            x0_flowopt_image = pipe.vae.decode(x0_flowopt_denorm, return_dict=False)[0].clamp(-1, 1)
        if gradio_app:
            x0_flowopt_image_pil = pipe.image_processor.postprocess(x0_flowopt_image)[0]
            history.append((x0_flowopt_image_pil, f"Iteration {flowopt_iter}"))
            yield history
        else:
            x0_flowopt_image = pipe.image_processor.postprocess(x0_flowopt_image, output_type="pt")

            flowopt_save_dir = os.path.join("saved_info", f"{exp_name}", "FLUX", f"eta={eta}", f"src_{src_prompt_txt}", f"tar_{tar_prompt_txt}")
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
