import numpy as np
import torch
import torchvision
import os
import piq

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

@torch.no_grad()
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

@torch.no_grad()
def calc_v_flux(
    pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance,
    text_ids, latent_image_ids, t,
):
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
def prep_input(pipe, scheduler, T_steps, x0_src, src_prompt, src_guidance_scale):
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

@torch.no_grad()
def uniinv(
    pipe, scheduler, timesteps, n_max, x0_src, src_prompt_embeds,
    src_pooled_prompt_embeds, src_guidance, src_text_ids, latent_src_image_ids,
):
    x_t = x0_src.clone()
    timesteps_inv = timesteps.flip(dims=(0,))[:-n_max] if n_max > 0 else timesteps.flip(dims=(0,))
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
    pipe, scheduler, T_steps, n_max, x0_src, src_prompt, src_guidance_scale,
):
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
        pipe, scheduler, timesteps, n_max, x0_src,
        src_prompt_embeds, src_pooled_prompt_embeds, src_guidance,
        src_text_ids, latent_src_image_ids,
    )

    return (
        x_t, x0_src, timesteps, latent_src_image_ids, orig_height, orig_width,
        src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids, src_guidance,
    )

@torch.no_grad()
def flux_denoise(
    pipe, scheduler, timesteps, n_max, x_t, prompt_embeds, pooled_prompt_embeds,
    guidance, text_ids, latent_image_ids,
):
    f_xt = x_t.clone()
    for _i, t in enumerate(timesteps[n_max:]):
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
    pipe, scheduler, T_steps, n_max, x0_src, src_prompt, src_guidance_scale,
    flowopt_iterations, eta, x_0, lpips_loss_fn, exp_name, src_prompt_txt,
):
    (
        x_t, x0_src, timesteps, latent_src_image_ids, orig_height, orig_width,
        src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids, src_guidance,
    ) = initialization(
        pipe, scheduler, T_steps, n_max, x0_src, src_prompt, src_guidance_scale
    )

    mse_array = np.zeros(flowopt_iterations + 1)
    lpips_array = np.zeros(flowopt_iterations + 1)
    ssim_array = np.zeros(flowopt_iterations + 1)

    j_star = x0_src.clone().to(torch.float32)
    for flowopt_iter in range(flowopt_iterations + 1):
        f_xt = flux_denoise(
            pipe, scheduler, timesteps, n_max, x_t,
            src_prompt_embeds, src_pooled_prompt_embeds, src_guidance,
            src_text_ids, latent_src_image_ids,
        )

        unpacked_f_xt = pipe._unpack_latents(f_xt, orig_height, orig_width, pipe.vae_scale_factor)
        f_xt_denorm = (unpacked_f_xt / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            f_xt_image = pipe.vae.decode(f_xt_denorm, return_dict=False)[0].clamp(-1, 1)
        f_xt_image = pipe.image_processor.postprocess(f_xt_image, output_type="pt")

        if flowopt_iter < flowopt_iterations:
            x_t = x_t.to(torch.float32)
            x_t = x_t - eta * (f_xt - j_star)
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
                    flowopt_save_dir, f"output_flowopt_iterations={flowopt_iter}_T_steps={T_steps}_n_max={n_max}_cfg={src_guidance_scale}.png",
                ),
            )

        mse_array[flowopt_iter] = torch.mean((255 * f_xt_image[0].detach().cpu().to(torch.float32) - 255 * x_0.detach().cpu().to(torch.float32))**2).numpy()
        lpips_array[flowopt_iter] = lpips_loss_fn(f_xt_image.detach().to(dtype=torch.float32), x_0.detach().to(dtype=torch.float32)).item()
        ssim_array[flowopt_iter] = piq.ssim(f_xt_image.detach().to(torch.float32), x_0.detach().to(torch.float32), data_range=1.0, reduction='sum').item()

    return (
        mse_array, lpips_array, ssim_array,
        enc_dec_img_ssim, enc_dec_img_lpips, enc_dec_img_mse, enc_dec_img_psnr,
    )

@torch.no_grad()
def flux_editing(
    pipe, scheduler, T_steps, n_max, x0_src, src_prompt, tar_prompt,
    src_guidance_scale, tar_guidance_scale, flowopt_iterations, eta,
    exp_name, src_prompt_txt, tar_prompt_txt,
):
    (
        x_t, x0_src, timesteps, latent_src_image_ids, orig_height, orig_width,
        _, _, _, _,
    ) = initialization(
        pipe, scheduler, T_steps, n_max, x0_src, src_prompt, src_guidance_scale
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

    j_star = x0_src.clone().to(torch.float32)
    for flowopt_iter in range(flowopt_iterations + 1):
        f_xt = flux_denoise(
            pipe, scheduler, timesteps, n_max, x_t,
            tar_prompt_embeds, pooled_tar_prompt_embeds, tar_guidance,
            tar_text_ids, latent_src_image_ids,
        )

        if flowopt_iter < flowopt_iterations:
            x_t = x_t.to(torch.float32)
            x_t = x_t - eta * (f_xt - j_star)
            x_t = x_t.to(x0_src.dtype)

        x0_flowopt = f_xt.clone()
        unpacked_x0_flowopt = pipe._unpack_latents(x0_flowopt, orig_height, orig_width, pipe.vae_scale_factor)
        x0_flowopt_denorm = (unpacked_x0_flowopt / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            x0_flowopt_image = pipe.vae.decode(x0_flowopt_denorm, return_dict=False)[0].clamp(-1, 1)
        x0_flowopt_image = pipe.image_processor.postprocess(x0_flowopt_image, output_type="pt")

        flowopt_save_dir = os.path.join("saved_info", f"{exp_name}", "FLUX", f"eta={eta}", f"src_{src_prompt_txt}", f"tar_{tar_prompt_txt}")
        os.makedirs(flowopt_save_dir, exist_ok=True)

        torchvision.utils.save_image(
            x0_flowopt_image,
            os.path.join(
                flowopt_save_dir, f"output_flowopt_iterations={flowopt_iter}_T_steps={T_steps}_n_max={n_max}_cfg={tar_guidance_scale}.png",
            ),
        )

        if flowopt_iter == flowopt_iterations:
            with open(f"{flowopt_save_dir}/prompts.txt", "w") as f:
                f.write(f"Source prompt: {src_prompt}\n")
                f.write(f"Target prompt: {tar_prompt}\n")
