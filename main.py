import os

import torch

from FCN import FCN
from constants import IMG_SIZE

workpath = "data"

image_paths, target_paths = [], []
for filename in os.listdir(os.path.join(workpath, "labels")):
    if filename.endswith(".png"):
        target_paths.append(os.path.join(workpath, "labels", filename))
        image_paths.append(os.path.join(workpath, "images", filename))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using GPU!", torch.cuda.get_device_name(None))
else:
    device = torch.device("cpu")
    print("Using CPU :(")

model = FCN(name="FCN_model", in_size=IMG_SIZE, device=device)
print(model)

model.train_model(image_paths, target_paths, workpath)
torch.save(model.state_dict(), os.path.join(workpath, "models", "dida_model.th"))
