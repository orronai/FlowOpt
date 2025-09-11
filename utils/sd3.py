import numpy as np
import torch
import torchvision
import os
import piq

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

@torch.no_grad()
def calc_v_sd3(
    pipe, latent_model_input, prompt_embeds,
    pooled_prompt_embeds, guidance_scale, t,
):
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

@torch.no_grad()
def uniinv(
    pipe, timesteps, n_start, x0_src, src_prompt_embeds_all,
    src_pooled_prompt_embeds_all, src_guidance_scale,
):
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
    pipe, scheduler, T_steps, n_start, x0_src,
    src_prompt, negative_prompt, src_guidance_scale,
):
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
    pipe, timesteps, n_start, x_t, prompt_embeds_all,
    pooled_prompt_embeds_all, guidance_scale,
):
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
    pipe, scheduler, T_steps, n_max, x0_src, src_prompt, negative_prompt, src_guidance_scale,
    flowopt_iterations, eta, x_0, lpips_loss_fn, exp_name, src_prompt_txt,
):
    n_start = T_steps - n_max
    (
        x_t, x0_src, timesteps, src_prompt_embeds_all, src_pooled_prompt_embeds_all,
    ) = initialization(
        pipe, scheduler, T_steps, n_start, x0_src, src_prompt, negative_prompt, src_guidance_scale,
    )

    mse_array = np.zeros(flowopt_iterations + 1)
    lpips_array = np.zeros(flowopt_iterations + 1)
    ssim_array = np.zeros(flowopt_iterations + 1)

    j_star = x0_src.clone().to(torch.float32)
    for flowopt_iter in range(flowopt_iterations + 1):
        f_xt = sd3_denoise(
            pipe, timesteps, n_start, x_t, src_prompt_embeds_all,
            src_pooled_prompt_embeds_all, src_guidance_scale,
        )

        f_xt_denorm = (f_xt / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            f_xt_image = pipe.vae.decode(f_xt_denorm, return_dict=False)[0].clamp(-1, 1)
        f_xt_image = pipe.image_processor.postprocess(f_xt_image, output_type="pt")

        if flowopt_iter < flowopt_iterations:
            x_t = x_t.to(torch.float32)
            x_t = x_t - eta * (f_xt - j_star)
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
                    flowopt_save_dir, f"output_flowopt_iterations={flowopt_iter}_T_steps={T_steps}_n_max={n_max}_cfg={src_guidance_scale}.png",
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
    pipe, scheduler, T_steps, n_max, x0_src, src_prompt, tar_prompt, negative_prompt,
    src_guidance_scale, tar_guidance_scale, flowopt_iterations, eta,
    exp_name, src_prompt_txt, tar_prompt_txt,
):
    n_start = T_steps - n_max
    x_t, x0_src, timesteps, _, _, = initialization(
        pipe, scheduler, T_steps, n_start, x0_src, src_prompt, negative_prompt, src_guidance_scale,
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

    j_star = x0_src.clone().to(torch.float32)
    for flowopt_iter in range(flowopt_iterations + 1):
        f_xt = sd3_denoise(
            pipe, timesteps, n_start, x_t, tar_prompt_embeds_all,
            tar_pooled_prompt_embeds_all, tar_guidance_scale,
        )

        if flowopt_iter < flowopt_iterations:
            x_t = x_t.to(torch.float32)
            x_t = x_t - eta * (f_xt - j_star)
            x_t = x_t.to(x0_src.dtype)

        x0_flowopt = f_xt.clone()
        x0_flowopt_denorm = (x0_flowopt / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            x0_flowopt_image = pipe.vae.decode(x0_flowopt_denorm, return_dict=False)[0].clamp(-1, 1)
        x0_flowopt_image = pipe.image_processor.postprocess(x0_flowopt_image, output_type="pt")

        flowopt_save_dir = os.path.join("saved_info", f"{exp_name}", "SD3", f"eta={eta}", f"src_{src_prompt_txt}", f"tar_{tar_prompt_txt}")
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
