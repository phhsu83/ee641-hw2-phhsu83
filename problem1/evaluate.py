"""
Analysis and evaluation experiments for trained GAN models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def _simple_letter_classifier(image):
    """
    Simple heuristic classifier for letters.
    This is a placeholder - in practice use a trained model.
    """
    # Extract simple features
    img = image.squeeze().cpu().numpy()
    
    # Use image statistics as features
    mean_val = img.mean()
    std_val = img.std()
    center_mass = img[10:18, 10:18].sum()
    
    # Simple hash to letter (this is just for demonstration)
    hash_val = int(mean_val * 100 + std_val * 10 + center_mass) % 26
    
    return hash_val

def interpolation_experiment(generator, device):
    """
    Interpolate between latent codes to generate smooth transitions.
    
    TODO:
    1. Find latent codes for specific letters (via optimization)
    2. Interpolate between them
    3. Visualize the path from A to Z
    """

    generator.eval()
    found = {}
    tries = 0

    with torch.no_grad():
        while len(found) < 2 and tries < 5000:

            z = torch.randn(1000, 100, device=device)
            imgs = (generator(z) + 1) / 2 


            preds = []
            for i in range(imgs.shape[0]):
                preds.append(_simple_letter_classifier(imgs[i]))

            # 依序檢查目標類別
            preds = np.array(preds)
            for target, name in [(0, "A"), (25, "Z")]:
                if name not in found:
                    idx = np.where(preds == target)[0]
                    if idx.size > 0:
                        found[name] = z[idx[0]].unsqueeze(0)

            tries += 1

        if "A" not in found or "Z" not in found:
            # 找不到就直接返回（也可選擇回傳 None/raise）
            return None

        z0, z1 = found["A"], found["Z"]

        # 建立插值路徑
        steps = 26
        t = torch.linspace(0, 1, steps, device=device)
        zs = (1 - t.view(-1,1)) * z0 + t.view(-1,1) * z1

        imgs = (generator(zs) + 1) / 2
        imgs = imgs.clamp(0, 1).cpu().numpy()      # [steps,1,28,28]

    # 繪圖
    cols = imgs.shape[0]
    fig, axes = plt.subplots(1, cols, figsize=(cols * 1.1, 1.2))
    if cols == 1:
        axes = [axes]
    for i in range(cols):
        axes[i].imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")
    plt.tight_layout()

    return fig



def style_consistency_experiment(conditional_generator, device):
    """
    Test if conditional GAN maintains style across letters.
    
    TODO:
    1. Fix a latent code z
    2. Generate all 26 letters with same z
    3. Measure style consistency
    """
    pass

def mode_recovery_experiment(generator_checkpoints):
    """
    Analyze how mode collapse progresses and potentially recovers.
    
    TODO:
    1. Load checkpoints from different epochs
    2. Measure mode coverage at each checkpoint
    3. Identify when specific letters disappear/reappear
    """
    pass