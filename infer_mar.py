import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
from models import mar
from models.vae import AutoencoderKL

# from util import download

torch.set_grad_enabled(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

model_type = "mar_huge"  # @param ["mar_base", "mar_large", "mar_huge"]
num_sampling_steps_diffloss = 100  # @param {type:"slider", min:1, max:1000, step:1}
if model_type == "mar_base":
    # download.download_pretrained_marb(overwrite=False)
    diffloss_d = 6
    diffloss_w = 1024
elif model_type == "mar_large":
    # download.download_pretrained_marl(overwrite=False)
    diffloss_d = 8
    diffloss_w = 1280
elif model_type == "mar_huge":
    # download.download_pretrained_marh(overwrite=False)
    diffloss_d = 12
    diffloss_w = 1536
else:
    raise NotImplementedError
model = mar.__dict__[model_type](
    buffer_size=64,
    diffloss_d=diffloss_d,
    diffloss_w=diffloss_w,
    num_sampling_steps=str(num_sampling_steps_diffloss)
).to(device)
state_dict = torch.load("pretrained_models/mar/{}/checkpoint-last.pth".format(model_type))["model_ema"]
model.load_state_dict(state_dict)
model.eval()  # important!
vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path="pretrained_models/vae/kl16.ckpt").cuda().eval()

seed = 42  # @param {type:"number"}
torch.manual_seed(seed)
np.random.seed(seed)
num_ar_steps = 64  # @param {type:"slider", min:1, max:256, step:1}
cfg_scale = 4  # @param {type:"slider", min:1, max:10, step:0.1}
cfg_schedule = "constant"  # @param ["linear", "constant"]
temperature = 0.5  # @param {type:"slider", min:0.9, max:1.1, step:0.01}
class_labels = 207, 360, 388, 113, 355, 980, 323, 979  # @param {type:"raw"}
samples_per_row = 4  # @param {type:"number"}

with torch.cuda.amp.autocast():
    sampled_tokens = model.sample_tokens(
        bsz=len(class_labels),
        num_iter=num_ar_steps,
        cfg=cfg_scale,
        cfg_schedule=cfg_schedule,
        labels=torch.Tensor(class_labels).long().cuda(),
        temperature=temperature, progress=True)
    sampled_images = vae.decode(sampled_tokens / 0.2325)

save_image(sampled_images, "sample.png", nrow=int(samples_per_row), normalize=True, value_range=(-1, 1))

# Convert the images to a NumPy array
sampled_images = sampled_images.float()
grid = make_grid(sampled_images, nrow=int(samples_per_row), normalize=True, value_range=(-1, 1)).cpu().numpy()

# Convert to HWC format and normalize to [0, 1]
grid = np.transpose(grid, (1, 2, 0))

# Display using Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(grid)
plt.axis('off')
plt.savefig("infer_mar.png", bbox_inches='tight')
plt.show()
