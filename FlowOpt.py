import argparse
import os
import random
import yaml

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from diffusers import FluxPipeline, StableDiffusion3Pipeline
from lpips import LPIPS

from utils.flux import flux_inversion, flux_editing
from utils.sd3 import sd3_inversion, sd3_editing


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_graph(y_values, x_label, y_label, save_path, h_line=None):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "cmu-serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })
    plt.style.use('tableau-colorblind10')

    plt.figure(figsize=(10, 5))
    plt.plot(y_values, label='FlowOpt', linewidth=2.5)
    if h_line is not None:
        plt.axhline(
            y=h_line, color='r', linestyle='--', linewidth=2.5,
            label='$\\mathcal{D}(\\mathcal{E}(\\boldsymbol{x}))$',
        )
    plt.legend(fontsize=20)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel(x_label, fontsize=24)
    plt.ylabel(y_label, fontsize=24)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_inversion_stats(
    enc_dec_img_ssim, enc_dec_img_lpips, enc_dec_img_mse, enc_dec_img_psnr,
    mse_array, lpips_array, ssim_array, exp_name, model_type, eta,
    max_iterations, src_guidance_scale, len_dataset_configs
):
    enc_dec_img_ssim /= len_dataset_configs
    enc_dec_img_lpips /= len_dataset_configs
    enc_dec_img_mse /= len_dataset_configs
    enc_dec_img_psnr /= len_dataset_configs
    # save to txt
    results_dir = f"saved_info/{exp_name}/{model_type}/eta={eta}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/enc_dec_results.txt", "a") as f:
        f.write(f"Exp name: {exp_name}\n")
        f.write(f"Avg Enc-Dec SSIM: {enc_dec_img_ssim:.8f}\n")
        f.write(f"Avg Enc-Dec LPIPS: {enc_dec_img_lpips:.8f}\n")
        f.write(f"Avg Enc-Dec MSE: {enc_dec_img_mse:.8f}\n")
        f.write(f"Avg Enc-Dec PSNR: {enc_dec_img_psnr:.8f}\n")
        f.write("\n")

    # save arrays
    psnr_array = 20 * np.log10(255 / np.sqrt(mse_array))
    np.save(f"{results_dir}/inversion_psnr.npy", psnr_array)
    np.save(f"{results_dir}/inversion_mse.npy", mse_array)
    np.save(f"{results_dir}/inversion_lpips.npy", lpips_array)
    np.save(f"{results_dir}/inversion_ssim.npy", ssim_array)

    psnr_array = np.mean(psnr_array, axis=0)
    rmse_array = np.sqrt(np.mean(mse_array, axis=0))
    lpips_array = np.mean(lpips_array, axis=0)
    ssim_array = np.mean(ssim_array, axis=0)

    # plot graphs
    plot_graph(
        psnr_array, "FlowOpt Iteration", "PSNR",
        os.path.join(results_dir, f"psnr_vs_iterations_flowopt_iterations_{max_iterations}_cfg={src_guidance_scale}.png"),
        h_line=enc_dec_img_psnr,
    )
    plot_graph(
        rmse_array, "FlowOpt Iteration", "RMSE",
        os.path.join(results_dir, f"rmse_vs_iterations_flowopt_iterations_{max_iterations}_cfg={src_guidance_scale}.png"),
        h_line=np.sqrt(enc_dec_img_mse),
    )
    plot_graph(
        lpips_array, "FlowOpt Iteration", "LPIPS",
        os.path.join(results_dir, f"lpips_vs_iterations_flowopt_iterations_{max_iterations}_cfg={src_guidance_scale}.png"),
        h_line=enc_dec_img_lpips,
    )
    plot_graph(
        ssim_array, "FlowOpt Iteration", "SSIM",
        os.path.join(results_dir, f"ssim_vs_iterations_flowopt_iterations_{max_iterations}_cfg={src_guidance_scale}.png"),
        h_line=enc_dec_img_ssim,
    )

@torch.no_grad()
def run_script(args):
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
                            flux_editing(
                                pipe, scheduler, T_steps, n_max, x0_src, src_prompt, tar_prompt,
                                src_guidance_scale, tar_guidance_scale, max_iterations, eta,
                                exp_name, src_prompt_txt, tar_prompt_txt,
                            )
                        else:  # SD3
                            sd3_editing(
                                pipe, scheduler, T_steps, n_max, x0_src, src_prompt, tar_prompt, negative_prompt,
                                src_guidance_scale, tar_guidance_scale, max_iterations, eta,
                                exp_name, src_prompt_txt, tar_prompt_txt,
                            )
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
