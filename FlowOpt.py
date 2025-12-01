import argparse
import os
import random

import numpy as np
import torch
import yaml
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from lpips import LPIPS
from PIL import Image
from tqdm import tqdm

from utils.flux import flux_editing, flux_inversion
from utils.inversion_utils import save_inversion_stats
from utils.sd3 import sd3_editing, sd3_inversion


def seed_everything(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def run_script(args: argparse.Namespace) -> None:
    """
    Run the main script for image inversion and editing using FLUX/SD3 models.

    Args:
        args (argparse.Namespace): Command line arguments containing device number and experiment YAML file path.
    """
    device = torch.device(f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu")
    exp_yaml = args.exp_yaml
    with open(exp_yaml) as file:
        exp_configs = yaml.load(file, Loader=yaml.FullLoader)

    model_type = exp_configs[0]["model_type"]
    if model_type == 'FLUX':
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16,
        )
    elif model_type == 'SD3':
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16,
        )
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")
    scheduler = pipe.scheduler
    pipe.to(device)

    for exp_dict in exp_configs:
        exp_name = exp_dict["exp_name"]
        T_steps = exp_dict["T_steps"]
        n_max = exp_dict["n_max"]
        max_iterations = exp_dict["max_iterations"]
        eta = exp_dict["eta"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict.get("tar_guidance_scale", 3.5)
        inversion_exp = exp_dict["inversion"]
        dataset_yaml = exp_dict["dataset_yaml"]
        seed = exp_dict["seed"]
        seed_everything(seed)

        with open(dataset_yaml) as file:
            dataset_configs = yaml.load(file, Loader=yaml.FullLoader)

        total_tar_prompts = 0
        for data_dict in dataset_configs:
            tar_prompts = data_dict["target_prompts"]
            total_tar_prompts += len(tar_prompts)
        for data_dict in dataset_configs:
            image_src_path = data_dict["init_img"]
            # check if image exists
            if not os.path.exists(image_src_path):
                print(f"Image {image_src_path} does not exist. Skipping.")
                continue

        if inversion_exp:
            mse_array = np.zeros((len(dataset_configs), max_iterations + 1))
            lpips_array = np.zeros((len(dataset_configs), max_iterations + 1))
            ssim_array = np.zeros((len(dataset_configs), max_iterations + 1))
            enc_dec_img_ssim = .0
            enc_dec_img_lpips = .0
            enc_dec_img_mse = .0
            enc_dec_img_psnr = .0

            lpips_loss_fn = LPIPS(net='vgg').to(device)
            lpips_loss_fn.eval()

        with tqdm(total=len(dataset_configs) if inversion_exp else total_tar_prompts) as pbar:
            for index, data_dict in enumerate(dataset_configs):
                src_prompt_txt = data_dict["init_img"].split("/")[-1].split(".")[0]
                src_prompt = data_dict["source_prompt"]
                tar_prompts = data_dict["target_prompts"]
                image_src_path = data_dict["init_img"]
                negative_prompt = ""
                image = Image.open(image_src_path)
                # crop image to have both dimensions divisibe by 16 - avoids issues with resizing
                image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
                image_src = pipe.image_processor.preprocess(image)
                x_0 = pipe.image_processor.postprocess(image_src, output_type="pt").to(device)
                # cast image to half precision
                image_src = image_src.to(device).half()
                with torch.autocast("cuda"), torch.inference_mode():
                    x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
                x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                # send to cuda
                x0_src = x0_src.to(device)

                if inversion_exp:
                    if model_type == 'FLUX':
                        (
                            mse_array[index, :], lpips_array[index, :], ssim_array[index, :],
                            enc_dec_img_ssim_cur, enc_dec_img_lpips_cur, enc_dec_img_mse_cur, enc_dec_img_psnr_cur,
                        ) = flux_inversion(
                            pipe, scheduler, T_steps, n_max, x0_src, src_prompt, src_guidance_scale,
                            max_iterations, eta, x_0, lpips_loss_fn, exp_name, src_prompt_txt,
                        )
                    else:  # SD3
                        (
                            mse_array[index, :], lpips_array[index, :], ssim_array[index, :],
                            enc_dec_img_ssim_cur, enc_dec_img_lpips_cur, enc_dec_img_mse_cur, enc_dec_img_psnr_cur,
                        ) =  sd3_inversion(
                            pipe, scheduler, T_steps, n_max, x0_src, src_prompt, negative_prompt, src_guidance_scale,
                            max_iterations, eta, x_0, lpips_loss_fn, exp_name, src_prompt_txt,
                        )
                    enc_dec_img_ssim += enc_dec_img_ssim_cur
                    enc_dec_img_lpips += enc_dec_img_lpips_cur
                    enc_dec_img_mse += enc_dec_img_mse_cur
                    enc_dec_img_psnr += enc_dec_img_psnr_cur
                    pbar.update(1)
                else:  # editing
                    for tar_idx, tar_prompt in enumerate(tar_prompts):
                        tar_prompt_txt = str(tar_idx)
                        if model_type == 'FLUX':
                            result = flux_editing(
                                pipe, scheduler, T_steps, n_max, x0_src, src_prompt, tar_prompt,
                                src_guidance_scale, tar_guidance_scale, max_iterations, eta,
                                exp_name, src_prompt_txt, tar_prompt_txt,
                            )
                            if result is not None:
                                for _ in result:
                                    pass
                        else:  # SD3
                            result = sd3_editing(
                                pipe, scheduler, T_steps, n_max, x0_src, src_prompt, tar_prompt, negative_prompt,
                                src_guidance_scale, tar_guidance_scale, max_iterations, eta,
                                exp_name, src_prompt_txt, tar_prompt_txt,
                            )
                            if result is not None:
                                for _ in result:
                                    pass
                        pbar.update(1)

            if inversion_exp:
                save_inversion_stats(
                    enc_dec_img_ssim, enc_dec_img_lpips, enc_dec_img_mse, enc_dec_img_psnr,
                    mse_array, lpips_array, ssim_array, exp_name, model_type, eta,
                    max_iterations, src_guidance_scale, len(dataset_configs)
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0, help="GPU device number")
    parser.add_argument("--exp_yaml", type=str, default="yaml_files/FLUX_editing_exp.yaml", help="Experiment YAML file")
    args = parser.parse_args()
    run_script(args)
