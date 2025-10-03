"""
GAN stabilization techniques to combat mode collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from collections import defaultdict
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt

from training_dynamics import analyze_mode_coverage, plot_mode_coverage_histogram, plot_alphabet_grid, visualize_mode_collapse

def train_gan_with_fix(generator, discriminator, data_loader, 
                       num_epochs=100, fix_type='feature_matching', device=None):
    """
    Train GAN with mode collapse mitigation techniques.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        fix_type: Stabilization method ('feature_matching', 'unrolled', 'minibatch')
        
    Returns:
        dict: Training history with metrics
    """
    
    if fix_type == 'feature_matching':
        # Feature matching: Match statistics of intermediate layers
        # instead of just final discriminator output
        
        def feature_matching_loss(real_images, fake_images, discriminator, real_labels):
            """
            TODO: Implement feature matching loss
            
            Extract intermediate features from discriminator
            Match mean statistics: ||E[f(x)] - E[f(G(z))]||²
            Use discriminator.features (before final classifier)
            """

            
            d_fake_for_g = discriminator((fake_images + 1) / 2)

            # 1) 對抗損失（騙過 D）
            g_adv = criterion(d_fake_for_g, real_labels)

            # 2) Feature Matching：取 D 的中間特徵
            with torch.no_grad():  # 真實特徵不回傳梯度到 D
                feat_real = discriminator.features(real_images)        # [B, C, H, W]
                feat_real_mean = feat_real.mean(dim=(0, 2, 3))         # [C]

            feat_fake = discriminator.features(fake_images)            # [B, C, H, W]
            feat_fake_mean = feat_fake.mean(dim=(0, 2, 3))             # [C]

            # L2 距離
            fm_loss = torch.mean((feat_fake_mean - feat_real_mean) ** 2)

            # 3) 合併損失
            lambda_fm = 0.05   # 可調：0.01～0.1 常見起點
            g_loss = g_adv + lambda_fm * fm_loss

            return g_loss

                        
    elif fix_type == 'unrolled':
        # Unrolled GANs: Look ahead k discriminator updates
        
        def unrolled_discriminator(discriminator, real_data, fake_data, k=5):
            """
            TODO: Implement k-step unrolled discriminator
            
            Create temporary discriminator copy
            Update it k times
            Compute generator loss through updated discriminator
            """
            pass
            
    elif fix_type == 'minibatch':
        # Minibatch discrimination: Let discriminator see batch statistics
        
        class MinibatchDiscrimination(nn.Module):
            """
            TODO: Add minibatch discrimination layer to discriminator
            
            Compute L2 distance between samples in batch
            Concatenate statistics to discriminator features
            """
            pass
    

    # Training loop with chosen fix
    # TODO: Implement modified training using selected technique



    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training history
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1):
        for batch_idx, (real_images, labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device) # [B, 1, 28, 28]
            
            # Labels for loss computation
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========== Train Discriminator ==========
            # TODO: Implement discriminator training step
            # 1. Zero gradients
            # 2. Forward pass on real images
            # 3. Compute real loss
            # 4. Generate fake images from random z
            # 5. Forward pass on fake images (detached)
            # 6. Compute fake loss
            # 7. Backward and optimize

            d_optimizer.zero_grad()

            # D(Real)
            d_real = discriminator(real_images)        # [B,1]
            d_real_loss = criterion(d_real, real_labels)

            # 產生假圖並給 D(Fake)
            z = torch.randn(batch_size, 100, device=device)

            fake_images = generator(z)
            d_fake = discriminator(((fake_images + 1) / 2).detach())

            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            
            # ========== Train Generator ==========
            # TODO: Implement generator training step
            # 1. Zero gradients
            # 2. Generate fake images
            # 3. Forward pass through discriminator
            # 4. Compute adversarial loss
            # 5. Backward and optimize

            g_optimizer.zero_grad()

            # 重新生成一批（或沿用同一批亦可）
            fake_images = generator(z)

            g_loss = feature_matching_loss(real_images, fake_images, discriminator, real_labels)
            
            g_loss.backward()
            g_optimizer.step()

            
            # Log metrics
            if batch_idx % 10 == 0:
                history['d_loss'].append(d_loss.item())
                history['g_loss'].append(g_loss.item())
                history['epoch'].append(epoch + batch_idx/len(data_loader))
        
        # Analyze mode collapse every 10 epochs
        if epoch % 10 == 0:
            mode_coverage = analyze_mode_coverage(generator, device)
            history['mode_coverage'].append(mode_coverage["coverage_score"])
            print(f"Epoch {epoch}: Mode coverage = {mode_coverage['coverage_score']:.2f}")

            #
            Path("results/visualizations").mkdir(parents=True, exist_ok=True)
            plot_mode_coverage_histogram(mode_coverage["letter_counts"], f"results/visualizations/mode_coverage_hist_{epoch}_fixed.png")


        #
        Path("results/visualizations").mkdir(parents=True, exist_ok=True)
        if epoch in [10, 30, 50, 100]:
            fig = plot_alphabet_grid(generator, device, z_dim=100, seed=None)
            # 存成 png
            fig.savefig(f"results/visualizations/letter_grids_{epoch}_fixed.png", dpi=200, bbox_inches="tight")
            # 用完記得關掉，避免記憶體累積
            plt.close(fig)

    
    #
    visualize_mode_collapse(history, "results/mode_collapse_analysis_fixed.png")

    return history