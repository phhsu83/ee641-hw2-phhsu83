"""
Training implementations for hierarchical VAE with posterior collapse prevention.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from pathlib import Path


def kl_annealing_schedule(epoch, method=None, total_epochs=100, cycles=5):
    """
    KL annealing schedule.

    Args:
        epoch:        當前的 epoch (從 0 開始)
        total_epochs: 總訓練 epochs，用來決定 annealing 範圍
        method:       'linear', 'sigmoid', 'cyclical'
        cycles:       週期式用，幾個循環

    Returns:
        beta: float ∈ [0,1]
    """
    if method == "linear":
        # 線性從 0 → 1
        return min(1.0, epoch / (0.3 * total_epochs))  # 前 30% epoch 升完

    elif method == "sigmoid":
        # sigmoid 曲線，前期上升慢，後期加速
        x = (epoch - total_epochs * 0.5) / (total_epochs * 0.1)
        return float(1 / (1 + np.exp(-x)))

    elif method == "cyclical":
        # 週期性 KL annealing：反覆從 0 → 1
        cycle_length = total_epochs // cycles
        cycle_pos = epoch % cycle_length
        return min(1.0, cycle_pos / (0.3 * cycle_length))

    else:
        raise ValueError(f"Unknown KL annealing method: {method}")


def temperature_annealing_schedule(epoch, method="linear", total_epochs=100, start=1.0, end=0.5):
    """
    Temperature annealing schedule.

    Args:
        epoch:        當前的 epoch
        total_epochs: 總訓練 epochs
        start:        初始 T
        end:          最小 T
        method:       'linear' 或 'exp'

    Returns:
        temperature: float
    """
    if method == "linear":
        # 線性下降
        t = epoch / total_epochs
        return max(end, start - (start - end) * t)

    elif method == "exp":
        # 指數衰減
        decay_rate = (end / start) ** (1 / total_epochs)
        return max(end, start * (decay_rate ** epoch))

    else:
        raise ValueError(f"Unknown temperature annealing method: {method}")


def train_hierarchical_vae(model, data_loader, num_epochs=100, device='cuda'):
    """
    Train hierarchical VAE with KL annealing and other tricks.
    
    Implements several techniques to prevent posterior collapse:
    1. KL annealing (gradual beta increase)
    2. Free bits (minimum KL per dimension)
    3. Temperature annealing for discrete outputs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # KL annealing schedule
    def kl_anneal_schedule(epoch):
        """
        TODO: Implement KL annealing schedule
        Start with beta ≈ 0, gradually increase to 1.0
        Consider cyclical annealing for better results
        """
        pass
    
    # Free bits threshold
    free_bits = 0.5  # Minimum nats per latent dimension
    
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        beta = kl_anneal_schedule(epoch)
        
        for batch_idx, patterns in enumerate(data_loader):
            patterns = patterns.to(device)
            
            # TODO: Implement training step
            # 1. Forward pass through hierarchical VAE
            # 2. Compute reconstruction loss
            # 3. Compute KL divergences (both levels)
            # 4. Apply free bits to prevent collapse
            # 5. Total loss = recon_loss + beta * kl_loss
            # 6. Backward and optimize
            
            pass
    
    return history


def plot_drum_pattern(pattern, title='Drum Pattern'):
    """
    Visualize a drum pattern as a piano roll.
    
    Args:
        pattern: Binary array [16, 9] or tensor
        title: Title for the plot
    
    Returns:
        figure: Matplotlib figure object
    """
    if torch.is_tensor(pattern):
        pattern = pattern.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    instruments = ['Kick', 'Snare', 'Hi-hat C', 'Hi-hat O', 
                  'Tom L', 'Tom H', 'Crash', 'Ride', 'Clap']
    
    # Create piano roll visualization
    for i in range(9):
        for j in range(16):
            if pattern[j, i] > 0.5:  # Threshold for binary
                ax.add_patch(plt.Rectangle((j, i), 1, 1, 
                                          facecolor='blue', 
                                          edgecolor='black',
                                          linewidth=0.5))
    
    # Grid
    for i in range(17):
        ax.axvline(i, color='gray', linewidth=0.5, alpha=0.5)
        if i % 4 == 0:
            ax.axvline(i, color='black', linewidth=1.5, alpha=0.7)
    
    for i in range(10):
        ax.axhline(i, color='gray', linewidth=0.5, alpha=0.5)
    
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_xticks(range(16))
    ax.set_xticklabels([str(i+1) for i in range(16)])
    ax.set_yticks(range(9))
    ax.set_yticklabels(instruments)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Instrument')
    ax.set_title(title)
    ax.invert_yaxis()
    
    return fig

'''
def sample_diverse_patterns(model, n_styles=5, n_variations=10, device='cuda'):
    """
    Generate diverse drum patterns using the hierarchy.
    
    TODO:
    1. Sample n_styles from z_high prior
    2. For each style, sample n_variations from conditional p(z_low|z_high)
    3. Decode to patterns
    4. Organize in grid showing style consistency
    """
'''

def sample_diverse_patterns(model, n_styles=5, n_variations=10,
                            temperature=1.0, threshold=0.5,
                            device='cuda'):
    """
    用層次式先驗生成多樣鼓點：
      1) 從 p(z_high)=N(0, I) 抽 n_styles 個風格向量
      2) 對每個風格，從 p(z_low|z_high) 抽 n_variations 個變化
      3) 解碼成 pattern logits → 機率/二值
      4) 以 grid 形式回傳

    Args:
        model: 你的 Hierarchical VAE，需包含：
               - model.z_high_dim
               - model.prior_low(z_high)  # 給出 [mu_p|logvar_p]
               - model.reparameterize(mu, logvar)
               - model.decode_hierarchy(z_high, z_low, temperature=...)
        n_styles:      風格數（高層樣本數）
        n_variations:  每個風格下的變化數（低層樣本數）
        temperature:   logits → 機率時的溫度縮放（推論用）
        threshold:     二值化門檻（>threshold 視為 1）
        return_probs:  True：同時回傳機率與二值；False：只回傳二值
        device:        裝置

    Returns:
        results: dict
          - patterns_bin:  [n_styles, n_variations, 16, 9]  (0/1)
          - patterns_prob: [n_styles, n_variations, 16, 9]  (若 return_probs=True)
          - z_high:        [n_styles, z_high_dim]
          - z_low:         [n_styles, n_variations, z_low_dim]
    """
    model.eval()
    z_high_dim = model.z_high_dim
    z_low_dim  = model.z_low_dim

    # 1) 從高層先驗抽風格向量（N(0,I)）
    z_high_styles = torch.randn(n_styles, z_high_dim, device=device)

    # 存放結果
    all_probs = []
    all_bins  = []
    all_z_low = []

    for i in range(n_styles):
        z_hi = z_high_styles[i]                           # [z_high_dim]
        z_hi_batch = z_hi.unsqueeze(0).repeat(n_variations, 1)  # [n_variations, z_high_dim]

        # 2) 條件先驗 p(z_low|z_high)：從 z_high 預測 (mu_p, logvar_p)，再抽樣 z_low
        prior_params = model.prior_low(z_hi_batch)        # [n_variations, 2*z_low_dim]
        mu_p, logvar_p = torch.chunk(prior_params, 2, dim=-1)
        z_low_batch = model.reparameterize(mu_p, logvar_p)  # [n_variations, z_low_dim]

        # 3) 解碼（批次解碼該風格下的多個變化）
        #    若你的 decode_hierarchy 會在 z_low=None 時自動從先驗抽樣，也可以改用 z_low=None。
        logits = model.decode_hierarchy(z_hi_batch, z_low=z_low_batch, temperature=1.0)
        # 有些版本的 decode_hierarchy 只回 logits；若回 tuple，取第一個即可
        if isinstance(logits, tuple):
            logits = logits[0]  # 兼容 earlier return (logits, z_low, mu_p, logvar_p)

        # 4) logits → 機率（溫度僅推論用，訓練時建議=1.0）
        probs = torch.sigmoid(logits / temperature)       # [n_variations, 16, 9]
        patt_bin = (probs > threshold).float()            # 二值化

        all_probs.append(probs.unsqueeze(0))              # [1, n_variations, 16, 9]
        all_bins.append(patt_bin.unsqueeze(0))            # [1, n_variations, 16, 9]
        all_z_low.append(z_low_batch.unsqueeze(0))        # [1, n_variations, z_low_dim]

        #
        Path("results/generated_patterns/").mkdir(parents=True, exist_ok=True)
        for j in range(len(patt_bin)):
            fig = plot_drum_pattern(patt_bin[j])
            fig.savefig(f"results/generated_patterns/samples_style_{i}_{j}.png", dpi=200)
            plt.close(fig)

    patterns_prob = torch.cat(all_probs, dim=0)
    patterns_bin  = torch.cat(all_bins,  dim=0)           # [n_styles, n_variations, 16, 9]
    z_low_grid    = torch.cat(all_z_low, dim=0)           # [n_styles, n_variations, z_low_dim]

    results = dict(
        patterns_bin=patterns_bin,
        patterns_prob = patterns_prob,
        z_high=z_high_styles,
        z_low=z_low_grid,
    )


    return results


def analyze_posterior_collapse(model, data_loader, device='cuda'):
    """
    Diagnose which latent dimensions are being used.
    
    TODO:
    1. Encode validation data
    2. Compute KL divergence per dimension
    3. Identify collapsed dimensions (KL ≈ 0)
    4. Return utilization statistics
    """
    pass