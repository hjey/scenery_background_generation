import os
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import pytorch_lightning as pl
from dataset import get_dataloader
from model import ADE20KOutdoorTrainer


# Set seed
pl.seed_everything(7777, workers=True)

# Set Dataloader
batch_size = 8
train_dataloader, valid_dataloader, test_dataloader = get_dataloader(
    batch_size, 
    mode='train',
    transform=[
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.5),
        A.ColorJitter(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.AdvancedBlur(p=0.2),
    ]
)

# Set Model
OUT_CLASSES = 151
arch = 'UPerNet'
backbone = "mit_b0"
learning_rate = 0.0001
model = ADE20KOutdoorTrainer(
    arch,
    backbone,
    in_channels=3,
    out_classes=OUT_CLASSES,
    learning_rate=learning_rate
)

# Set Trainer
EPOCHS = 300
model_name = 'UPerNet'
version_name = "mit_b0_main"
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    log_every_n_steps=1,
    devices=[0],
    callbacks=[pl.callbacks.ModelCheckpoint(
        filename='{valid_dataset_iou:.2f}-{epoch}',
        monitor='valid_dataset_iou', 
        save_top_k=1, 
        mode='max'
        )],
    logger=pl.loggers.TensorBoardLogger('outdoor', name=model_name, version=version_name), # name = "Default Folder", version = "Logger Name"
)

# Training / Validation / Test
# trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
BEST_PT_DEFAULT_DIR = f'/home/work/Backscape/project/outdoor/{model_name}/{version_name}'
best_ckpt = sorted(glob(f'{BEST_PT_DEFAULT_DIR}/checkpoints/*'))[-1]

trainer.validate(model, dataloaders=valid_dataloader, ckpt_path=best_ckpt)
trainer.test(model, dataloaders=test_dataloader, ckpt_path=best_ckpt)


# Predict
os.makedirs(f'{BEST_PT_DEFAULT_DIR}/predict_sample', exist_ok=True)
os.makedirs(f'{BEST_PT_DEFAULT_DIR}/predict_segmentmap_only', exist_ok=True)
_, _, test_dataloader = get_dataloader(1, mode='predict')
predict_results = trainer.predict(
    model=model,
    dataloaders=test_dataloader, 
    ckpt_path=best_ckpt
)

for idx, data in enumerate(test_dataloader):
    target = np.transpose(data["image"][0].numpy(), (1, 2, 0))
    output = predict_results[int(idx)][0].numpy()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(target)
    plt.title('target')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(output)
    plt.title('output')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(target)
    plt.imshow(output, alpha=0.6)
    plt.axis('off')
    plt.tight_layout()
    
    idx = str(idx)
    if   len(idx) == 1: idx = f'00{idx}'
    elif len(idx) == 2: idx = f'0{idx}'
    plt.savefig(f"{BEST_PT_DEFAULT_DIR}/predict_sample/{idx}.png", bbox_inches='tight', pad_inches=0)
    cv2.imwrite(f"{BEST_PT_DEFAULT_DIR}/predict_segmentmap_only/{idx}.png", output)

    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure