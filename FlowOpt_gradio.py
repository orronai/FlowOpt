import random
from typing import Tuple

import gradio as gr
import numpy as np
import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from PIL import Image

from utils.flux import flux_editing
from utils.sd3 import sd3_editing

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe_sd3 = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
)
pipe_flux = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
)


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

def on_T_steps_change(T_steps: int, n_max: int) -> gr.update:
    """
    Update the maximum and value of the n_max slider based on T_steps.

    Args:
        T_steps (int): The current value of the T_steps slider.
        n_max (int): The current value of the n_max slider.
    Returns:
        gr.update: An update object to modify the n_max slider.
    """
    # If n_max > T_steps, clamp it down to T_steps
    new_value = min(n_max, T_steps)
    return gr.update(maximum=T_steps, value=new_value)

def on_model_change(model_type: str) -> Tuple[int, int, float]:
    if model_type == 'SD3':
        T_steps_value = 15
        n_max_value = 12
        eta_value = 0.01
    elif model_type == 'FLUX':
        T_steps_value = 15
        n_max_value = 13
        eta_value = 0.0025
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    return T_steps_value, n_max_value, eta_value

def get_examples():
    case = [
        ["gradio_examples/inputs/corgi_walking.png", "FLUX", 15, 13, 0.0025, 7, "A cute brown and white dog walking on a sidewalk near a body of water. The dog is wearing a pink vest, adding a touch of color to the scene.", "A cute brown and white guinea pig walking on a sidewalk near a body of water. The guinea pig is wearing a pink vest, adding a touch of color to the scene.", 1.0, 3.5, [(f"gradio_examples/outputs/corgi_walking/guinea_pig/flux_iterations={i}.png", f"Iteration {i}") for i in range(8)]],
        ["gradio_examples/inputs/puppies.png", "FLUX", 15, 13, 0.0025, 7, "Two adorable golden retriever puppies sitting in a grassy field. They are positioned close to each other, with one dog on the left and the other on the right. Both dogs have their mouths open, possibly panting.", "Two adorable crochet golden retriever puppies sitting in a grassy field. They are positioned close to each other, with one dog on the left and the other on the right. Both dogs have their mouths open, possibly panting or enjoying the outdoor environment.", 1.0, 3.5, [(f"gradio_examples/outputs/puppies/crochet/flux_iterations={i}.png", f"Iteration {i}") for i in range(8)]],
        ["gradio_examples/inputs/iguana.png", "FLUX", 15, 13, 0.0025, 7, "A large orange lizard sitting on a rock near the ocean. The lizard is positioned in the center of the scene, with the ocean waves visible in the background. The rock is located close to the water, providing a picturesque setting for the lizard''s resting spot.", "A large lizard made out of lego bricks sitting on a rock near the ocean. The lizard is positioned in the center of the scene, with the ocean waves visible in the background. The rock is located close to the water, providing a picturesque setting for the lizard''s resting spot.", 1.0, 3.5, [(f"gradio_examples/outputs/iguana/lego_bricks/flux_iterations={i}.png", f"Iteration {i}") for i in range(8)]],
        ["gradio_examples/inputs/cow_grass2.png", "FLUX", 15, 13, 0.0025, 5, "A large brown and white cow standing in a grassy field. The cow is positioned towards the center of the scene. The field is lush and green, providing a perfect environment for the cow to graze.", "A large cow made out of flowers standing in a grassy field. The flower cow is positioned towards the center of the scene. The field is lush and green, providing a perfect environment for the cow to graze.", 1.0, 3.5, [(f"gradio_examples/outputs/cow_grass2/flowers/flux_iterations={i}.png", f"Iteration {i}") for i in range(6)]],
        ["gradio_examples/inputs/cow_grass2.png", "SD3", 15, 12, 0.01, 8, "A large brown and white cow standing in a grassy field. The cow is positioned towards the center of the scene. The field is lush and green, providing a perfect environment for the cow to graze.", "A large cow made out of wooden blocks standing in a grassy field. The wooden block cow is positioned towards the center of the scene. The field is lush and green, providing a perfect environment for the cow to graze.", 1.0, 3.5, [(f"gradio_examples/outputs/cow_grass2/wooden_blocks/sd3_iterations={i}.png", f"Iteration {i}") for i in range(9)]],
        ["gradio_examples/inputs/cat_fridge.png", "SD3", 15, 12, 0.01, 8, "A cat sitting on top of a counter in a store. The cat is positioned towards the right side of the counter, and it appears to be looking at the camera. The store has a variety of items displayed, including several bottles scattered around the counter.", "A cat sitting on top of a counter in a store, with the cat and counter crafted using origami folded paper art techniques. The cat has a delicate and intricate appearance, with paper folds used to create its fur and features. The store has a variety of items displayed, including several bottles scattered around the counter.", 1.0, 3.5, [(f"gradio_examples/outputs/cat_fridge/origami/sd3_iterations={i}.png", f"Iteration {i}") for i in range(9)]],
        ["gradio_examples/inputs/cat.png", "SD3", 15, 12, 0.01, 6, "A small, fluffy kitten sitting in a grassy field. The kitten is positioned in the center of the scene, surrounded by a field. The kitten appears to be looking at something in the field.", "A small puppy sitting in a grassy field. The puppy is positioned in the center of the scene, surrounded by a field. The puppy appears to be looking at something in the field.", 1.0, 3.5, [(f"gradio_examples/outputs/cat/puppy/sd3_iterations={i}.png", f"Iteration {i}") for i in range(7)]],
        ["gradio_examples/inputs/wolf_grass.png", "SD3", 15, 12, 0.01, 4, "A wolf standing in a grassy field with yellow flowers. The wolf is positioned towards the center of the scene, and its body is facing the camera. The field is filled with grass, and the yellow flowers are scattered throughout the area.", "A baby deer standing in a grassy field with yellow flowers. The baby deer is positioned towards the center of the scene, and its body is facing the camera. The field is filled with grass, and the yellow flowers are scattered throughout the area.", 1.0, 3.5, [(f"gradio_examples/outputs/wolf_grass/deer/sd3_iterations={i}.png", f"Iteration {i}") for i in range(5)]],
    ]
    return case

def FlowOpt_run(
    image_src_val: str, model_type_val: str, T_steps_val: int,
    n_max_val: int, eta_val: float, flowopt_iterations_val: int,
    src_prompt_val: str, tar_prompt_val: str,
    src_guidance_scale_val: float, tar_guidance_scale_val: float,
):
    if not len(src_prompt_val):
        raise gr.Error("Source prompt cannot be empty")
    if not len(tar_prompt_val):
        raise gr.Error("Target prompt cannot be empty")

    if model_type_val == 'FLUX':
        pipe = pipe_flux.to(device)
    elif model_type_val == 'SD3':
        pipe = pipe_sd3.to(device)
    else:
        raise NotImplementedError(f"Model type {model_type_val} not implemented")

    scheduler = pipe.scheduler

    # set seed
    seed = 1024
    seed_everything(seed)
    # load image
    image = Image.open(image_src_val)
    # crop image to have both dimensions divisibe by 16 - avoids issues with resizing
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    image_src_val = pipe.image_processor.preprocess(image)

    # cast image to half precision
    image_src_val = image_src_val.to(device).half()
    with torch.autocast("cuda"), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(image_src_val).latent_dist.mode()
    x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    # send to cuda
    x0_src = x0_src.to(device)
    negative_prompt =  ""  # (SD3)

    if model_type_val == 'SD3':
        yield from sd3_editing(
            pipe, scheduler, T_steps_val, n_max_val, x0_src,
            src_prompt_val, tar_prompt_val, negative_prompt,
            src_guidance_scale_val, tar_guidance_scale_val,
            flowopt_iterations_val, eta_val, gradio_app=True,
        )
    elif model_type_val == 'FLUX':
        yield from flux_editing(
            pipe, scheduler, T_steps_val, n_max_val, x0_src,
            src_prompt_val, tar_prompt_val,
            src_guidance_scale_val, tar_guidance_scale_val,
            flowopt_iterations_val, eta_val, gradio_app=True,
        )
    else:
        raise NotImplementedError(f"Sampler type {model_type_val} not implemented")


intro = """
<h1 style="font-weight: 1000; text-align: center; margin: 0px;">FlowOpt: Fast Optimization Through Whole Flow Processes for Training-Free Editing</h1>
<h3 style="margin-bottom: 10px; text-align: center;">
    <a href="https://arxiv.org/abs/2510.22010">[Paper]</a>&nbsp;|&nbsp;
    <a href="https://orronai.github.io/FlowOpt/">[Project Page]</a>&nbsp;|&nbsp;
    <a href="https://github.com/orronai/FlowOpt">[Code]</a>&nbsp;|&nbsp;
    <a href="https://huggingface.co/spaces/orronai/FlowOpt">[Space]</a>
</h3>
<br> 🎨 Edit your image using FlowOpt for Flow models! Upload an image, add a description of it, and specify the edits you want to make.
<h3>Notes:</h3>
<ol>
  <li>We use FLUX.1 dev and SD3 for the demo. The models are large and may take a while to load.</li>
  <li>We recommend 1024x1024 images for the best results. If the input images are too large, there may be out-of-memory errors. For other resolutions, we encourage you to find a suitable set of hyperparameters.</li>
  <li>Default hyperparameters for each model used in the paper are provided as examples.</li>
</ol>  
"""

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}

#gallery-image img {
    width: 100%;        /* match column width */
    height: auto;       /* preserve aspect ratio */
    object-fit: contain;
}

/* Hide thumbnails by default */
#gallery-image .gallery-container .preview .thumbnails {
    opacity: 0;
    transition: opacity 0.3s ease-in-out;
}

/* Show thumbnails only when hovering over the gallery */
#gallery-image:hover .gallery-container .preview .thumbnails {
    opacity: 1;
}

/* Hide caption by default */
.gallery-container .preview .caption {
    opacity: 0;
    transition: opacity 0.3s ease-in-out;
}

/* Show caption when hovering over #gallery-image */
#gallery-image:hover .gallery-container .preview .caption {
    opacity: 1;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(intro)

        with gr.Row():
            with gr.Column():
                image_src = gr.Image(type="filepath", label="Source Image", value="gradio_examples/inputs/corgi_walking.png",)
                src_prompt = gr.Textbox(lines=2, label="Source Prompt", value="A cute brown and white dog walking on a sidewalk near a body of water. The dog is wearing a pink vest, adding a touch of color to the scene.")
                tar_prompt = gr.Textbox(lines=2, label="Target Prompt", value="A cute brown and white dog walking on a sidewalk near a body of water. The dog is wearing a pink vest, adding a touch of color to the scene. The dog and sidewalk are constructed entirely out of Lego bricks, showcasing a blocky and geometric appearance.")
                submit_button = gr.Button("Run FlowOpt", variant="primary")

                with gr.Row():
                    model_type = gr.Dropdown(["FLUX", "SD3"], label="Model Type", value="FLUX")
                    T_steps = gr.Slider(value=15, minimum=10, maximum=50, step=1, label="Total Steps", info="Total number of discretization steps.")
                    n_max = gr.Slider(value=13, minimum=1, maximum=15, step=1, label="n_max", info="Control the strength of the edit.")
                    eta = gr.Slider(value=0.0025, minimum=0.0001, maximum=0.05, label="eta", info="Control the optimization step-size (η).")
                    flowopt_iterations = gr.Number(value=8, minimum=1, maximum=15, label="flowopt_iterations", info="Max number of FlowOpt iterations (N).")

            with gr.Column():
                image_tar = gr.Gallery(
                    label="Outputs", show_label=True, format="png",
                    columns=[3], rows=[3], height="auto", elem_id="gallery-image",
                )
        with gr.Accordion(label="Advanced Settings", open=False):
            src_guidance_scale = gr.Slider(value=1.0, minimum=0.0, maximum=15.0, label="src_guidance_scale", info="Source prompt CFG scale.")
            tar_guidance_scale = gr.Slider(value=3.5, minimum=1.0, maximum=15.0, label="tar_guidance_scale", info="Target prompt CFG scale.")

    submit_button.click(
        fn=FlowOpt_run, 
        inputs=[
            image_src, model_type, T_steps, n_max, eta, flowopt_iterations,
            src_prompt, tar_prompt, src_guidance_scale, tar_guidance_scale,
        ],
        outputs=[image_tar],
    )

    gr.Examples(
        label="Examples",
        examples=get_examples(),
        inputs=[
            image_src, model_type, T_steps, n_max, eta,
            flowopt_iterations, src_prompt, tar_prompt,
            src_guidance_scale, tar_guidance_scale, image_tar,
        ],
        outputs=[image_tar],
    )

    model_type.input(fn=on_model_change, inputs=[model_type], outputs=[T_steps, n_max, eta])
    T_steps.change(fn=on_T_steps_change, inputs=[T_steps, n_max], outputs=[n_max])

demo.queue()
demo.launch()
