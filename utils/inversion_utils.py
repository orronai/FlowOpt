import os

import matplotlib.pyplot as plt
import numpy as np


def plot_graph(
    y_values: np.ndarray, x_label: str, y_label: str,
    save_path: str, h_line: float = None,
) -> None:
    """
    Plot a graph of y_values vs. iterations and save it to a file.

    Args:
        y_values (np.ndarray): The y-values to plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        save_path (str): The path to save the plot image.
        h_line (float, optional): A horizontal line value to plot for reference. Defaults to None.
    """
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
    enc_dec_img_ssim: float, enc_dec_img_lpips: float,
    enc_dec_img_mse: float, enc_dec_img_psnr: float,
    mse_array: np.ndarray, lpips_array: np.ndarray, ssim_array: np.ndarray,
    exp_name: str, model_type: str, eta: float,
    max_iterations: int, src_guidance_scale: float, len_dataset_configs: int,
) -> None:
    """
    Save inversion statistics to a text file and plot graphs.

    Args:
        enc_dec_img_ssim (float): The average SSIM of the encoded-decoded images.
        enc_dec_img_lpips (float): The average LPIPS of the encoded-decoded images.
        enc_dec_img_mse (float): The average MSE of the encoded-decoded images.
        enc_dec_img_psnr (float): The average PSNR of the encoded-decoded images.
        mse_array (np.ndarray): The MSE values over iterations.
        lpips_array (np.ndarray): The LPIPS values over iterations.
        ssim_array (np.ndarray): The SSIM values over iterations.
        exp_name (str): The name of the experiment for saving results.
        model_type (str): The type of model used (e.g., 'FLUX', 'SD3').
        eta (float): The step size for the FlowOpt update.
        max_iterations (int): The maximum number of FlowOpt iterations.
        src_guidance_scale (float): The guidance scale used for the source prompt.
        len_dataset_configs (int): The number of images in the dataset.
    """
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
        os.path.join(
            results_dir,
            f"psnr_vs_iterations_flowopt_iterations_{max_iterations}_cfg={src_guidance_scale}.png",
        ),
        h_line=enc_dec_img_psnr,
    )
    plot_graph(
        rmse_array, "FlowOpt Iteration", "RMSE",
        os.path.join(
            results_dir,
            f"rmse_vs_iterations_flowopt_iterations_{max_iterations}_cfg={src_guidance_scale}.png",
        ),
        h_line=np.sqrt(enc_dec_img_mse),
    )
    plot_graph(
        lpips_array, "FlowOpt Iteration", "LPIPS",
        os.path.join(
            results_dir,
            f"lpips_vs_iterations_flowopt_iterations_{max_iterations}_cfg={src_guidance_scale}.png",
        ),
        h_line=enc_dec_img_lpips,
    )
    plot_graph(
        ssim_array, "FlowOpt Iteration", "SSIM",
        os.path.join(
            results_dir,
            f"ssim_vs_iterations_flowopt_iterations_{max_iterations}_cfg={src_guidance_scale}.png",
        ),
        h_line=enc_dec_img_ssim,
    )
