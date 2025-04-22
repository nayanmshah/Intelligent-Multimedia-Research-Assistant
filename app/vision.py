from diffusers import StableDiffusionPipeline
import torch

def generate_visual_summary(prompt):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to("cuda")
    image = pipe(prompt).images[0]
    image.save("summary_image.png")
    return "summary_image.png"