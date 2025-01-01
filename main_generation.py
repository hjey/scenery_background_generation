import os
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import pytorch_lightning as pl
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from dataset import get_dataloader
from model import ADE20KOutdoorTrainer
from palette import palette


model_name = 'UPerNet'
version_name = 'mit_b0_main'

# Set Dataloader
_, _, test_dataloader = get_dataloader(1, mode='predict')

# ────────Segmentation────────
# Set Model
OUT_CLASSES = 151
arch = 'UPerNet'
backbone = "mit_b0"
learning_rate = 0.0001
pretrained_model = ADE20KOutdoorTrainer(
    arch,
    backbone,
    in_channels=3,
    out_classes=OUT_CLASSES,
    learning_rate=learning_rate
)

# Set Trainer
trainer = pl.Trainer(
    log_every_n_steps=1,
    devices=[0],
)

# Predict
BEST_PT_DEFAULT_DIR = f'/home/work/Backscape/project/outdoor/{model_name}/{version_name}'
best_ckpt = sorted(glob(f'{BEST_PT_DEFAULT_DIR}/checkpoints/*'))[-1]
predict_results = trainer.predict(
    model=pretrained_model, 
    dataloaders=test_dataloader, 
    ckpt_path=best_ckpt
)

# # Save Segment Image
# os.makedirs(f'{BEST_PT_DEFAULT_DIR}/predict_sample', exist_ok=True)
# os.makedirs(f'{BEST_PT_DEFAULT_DIR}/predict_segmentmap_only', exist_ok=True)
# for idx, data in enumerate(test_dataloader):
#     target = np.transpose(data["image"][0].numpy(), (1, 2, 0))
#     output = predict_results[int(idx)][0].numpy()

#     plt.figure(figsize=(10, 10))
#     plt.subplot(1, 3, 1)
#     plt.imshow(target)
#     plt.title('target')
#     plt.axis('off')
#     plt.subplot(1, 3, 2)
#     plt.imshow(output)
#     plt.title('output')
#     plt.axis('off')
#     plt.subplot(1, 3, 3)
#     plt.imshow(target)
#     plt.imshow(output, alpha=0.6)
#     plt.axis('off')
#     plt.tight_layout()
    
#     idx = str(idx)
#     if   len(idx) == 1: idx = f'00{idx}'
#     elif len(idx) == 2: idx = f'0{idx}'
#     plt.savefig(f"{BEST_PT_DEFAULT_DIR}/predict_sample/{idx}.png", bbox_inches='tight', pad_inches=0)
#     cv2.imwrite(f"{BEST_PT_DEFAULT_DIR}/predict_segmentmap_only/{idx}.png", output)

#     plt.cla()   # clear the current axes
#     plt.clf()   # clear the current figure
#     plt.close() # closes the current figure

# ────────Generation────────
# Set Model
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-seg", 
        torch_dtype=torch.float16
    ), 
    safety_checker=None, 
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Generate New Image
for idx, data in enumerate(test_dataloader):
    target = np.transpose(data["image"][0].numpy(), (1, 2, 0))
    output = predict_results[idx][0]
    color_output = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_output[output == label, :] = color

    output = output.numpy()
    idx = str(idx)
    if   len(idx) == 1: idx = f'00{idx}'
    elif len(idx) == 2: idx = f'0{idx}'

    os.makedirs(f'{BEST_PT_DEFAULT_DIR}/generate_sample/{idx}',     exist_ok=True)
    os.makedirs(f'{BEST_PT_DEFAULT_DIR}/generate_image_only/{idx}', exist_ok=True)
    for prompt in [
        'Snowy',
        'Swamp',
        'Van Gogh Style',
        'War',
        'Rainbow',
        'Dreams Come True',
    ]:
        generated_image = pipe(
            prompt, 
            Image.fromarray(color_output), 
            num_inference_steps=20
        ).images[0]

        # Save Sample
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(target)
        plt.title('target')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(target)
        plt.imshow(output, alpha=0.6)
        plt.title('segmentation results')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(generated_image)
        plt.title(f'{prompt}')
        plt.axis('off')
        plt.tight_layout()

        title = "no_prompt" if prompt == "" else prompt
        plt.savefig(f"{BEST_PT_DEFAULT_DIR}/generate_sample/{idx}/{title}.png", bbox_inches='tight', pad_inches=0)
        generated_image.save(f'{BEST_PT_DEFAULT_DIR}/generate_image_only/{idx}/{title}.png')

        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

