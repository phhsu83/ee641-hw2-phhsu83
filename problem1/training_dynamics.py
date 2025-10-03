"""
GAN training implementation with mode collapse analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict

from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

def train_gan(generator, discriminator, data_loader, num_epochs=100, device=None):
    """
    Standard GAN training implementation.
    
    Uses vanilla GAN objective which typically exhibits mode collapse.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device for computation
        
    Returns:
        dict: Training history and metrics
    """
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
            d_fake_for_g = discriminator((fake_images + 1) / 2)

            # 讓 D 認為是假圖為真（騙過 D）
            g_loss = criterion(d_fake_for_g, real_labels)
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
            plot_mode_coverage_histogram(mode_coverage["letter_counts"], f"results/visualizations/mode_coverage_hist_{epoch}.png")


        #
        Path("results/visualizations").mkdir(parents=True, exist_ok=True)
        if epoch in [10, 30, 50, 100]:
            fig = plot_alphabet_grid(generator, device, z_dim=100, seed=None)
            # 存成 png
            fig.savefig(f"results/visualizations/letter_grids_{epoch}.png", dpi=200, bbox_inches="tight")
            # 用完記得關掉，避免記憶體累積
            plt.close(fig)

    
    #
    visualize_mode_collapse(history, "results/mode_collapse_analysis.png")

    return history


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

def mode_coverage_score(generated_samples, classifier_fn=None):
    """
    Measure how many different letter modes the GAN generates.
    
    Args:
        generated_samples: Tensor of generated images [N, 1, 28, 28]
        classifier_fn: Function to classify letters (if None, use simple heuristic)
    
    Returns:
        Dict with:
            - coverage_score: Number of unique letters / 26
            - letter_counts: Dict of letter -> count
            - missing_letters: List of letters not generated
    """
    if classifier_fn is None:
        # Simple heuristic classifier based on image statistics
        # In practice, students would use a pre-trained classifier
        classifier_fn = _simple_letter_classifier
    
    # Classify all samples
    predictions = []
    for i in range(generated_samples.shape[0]):
        letter = classifier_fn(generated_samples[i])
        predictions.append(letter)
    
    # Count unique letters
    letter_counts = Counter(predictions)
    unique_letters = set(letter_counts.keys())
    all_letters = set(range(26))  # 0-25 for A-Z
    missing_letters = sorted(all_letters - unique_letters)
    
    coverage_score = len(unique_letters) / 26.0
    
    return {
        'coverage_score': coverage_score,
        'letter_counts': dict(letter_counts),
        'missing_letters': missing_letters,
        'n_unique': len(unique_letters)
    }



def analyze_mode_coverage(generator, device, n_samples=1000):
    """
    Measure mode coverage by counting unique letters in generated samples.
    
    Args:
        generator: Trained generator network
        device: Device for computation
        n_samples: Number of samples to generate
        
    Returns:
        float: Coverage score (unique letters / 26)
    """
    # TODO: Generate n_samples images
    # Use provided letter classifier to identify generated letters
    # Count unique letters produced
    # Return coverage score (0 to 1)
    
    generator.eval()

    with torch.no_grad():
        
        z = torch.randn(n_samples, 100, device=device)

        fake_imgs = generator(z)

    # 這裡直接調用你的函式
    return mode_coverage_score((fake_imgs + 1) / 2) 



def visualize_mode_collapse(history, save_path):
    """
    Visualize mode collapse progression over training.
    
    Args:
        history: Training metrics dictionary
        save_path: Output path for visualization
    """
    # TODO: Plot mode coverage over time
    # Show which letters survive and which disappear


    if 'mode_coverage' not in history or len(history['mode_coverage']) == 0:
        raise ValueError("history['mode_coverage']: need value")

    cov = np.asarray(history['mode_coverage'], dtype=float)
    # 對應的 x 軸：優先使用提供的 epoch，否則用索引
    '''
    if 'epoch' in history and len(history['epoch']) >= len(cov):
        x = np.asarray(history['epoch'][:len(cov)], dtype=float)
    else:
        x = np.arange(len(cov), dtype=float)
    '''
    x = list(range(10, (len(cov) + 1) * 10, 10))


    # 繪圖
    plt.figure(figsize=(6, 4))
    plt.plot(x, cov, marker='o', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Mode coverage (unique letters / 26)")
    plt.ylim(0.0, 1.05)
    plt.title("Mode coverage over training")
    plt.grid(True, linestyle='--', alpha=0.4)

    # 註記最後一點
    plt.annotate(f"{cov[-1]:.2f}",
                 xy=(x[-1], cov[-1]),
                 xytext=(5, 5),
                 textcoords="offset points")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_mode_coverage_histogram(letter_counts, save_path=None):

    
    sorted_dict = dict(sorted(letter_counts.items()))

    letters = [chr(65+i) for i in range(26)]
    ys = [sorted_dict.get(i, 0) for i in range(26)]
    plt.figure(figsize=(10,6))
    plt.bar(range(26), ys)
    plt.xticks(range(26), letters)
    plt.ylabel("count")
    plt.title("Mode coverage histogram")
    plt.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close() 



def plot_alphabet_grid(generator, device, z_dim=100, seed=None):
    """
    Generate and plot a grid of all 26 letters.
    
    Args:
        generator: Trained generator model
        device: Device to run on
        z_dim: Dimension of latent space
        seed: Random seed for reproducibility
    
    Returns:
        figure: Matplotlib figure object
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    generator.eval()
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(4, 7, figure=fig)
    
    with torch.no_grad():
        for i in range(26):
            ax = fig.add_subplot(gs[i // 7, i % 7])
            
            # Generate random z
            z = torch.randn(1, z_dim).to(device)
            
            # Generate image
            if hasattr(generator, 'conditional') and generator.conditional:
                # Conditional GAN - provide letter label
                label = torch.zeros(1, 26).to(device)
                label[0, i] = 1
                fake_img = generator(z, label).squeeze().cpu()
            else:
                # Unconditional - just generate
                fake_img = generator(z).squeeze().cpu()
            
            # Convert from [-1, 1] to [0, 1] for display
            fake_img = (fake_img + 1) / 2
            
            ax.imshow(fake_img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(chr(65 + i), fontsize=12)
            ax.axis('off')
    
    plt.suptitle('Generated Alphabet', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig