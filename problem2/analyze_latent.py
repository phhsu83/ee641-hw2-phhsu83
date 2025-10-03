"""
Latent space analysis tools for hierarchical VAE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json

from pathlib import Path
from training_utils import plot_drum_pattern

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def plot_latent_space_2d(latent_codes, labels=None, title='Latent Space'):
    """
    Plot 2D visualization of latent space.
    
    Args:
        latent_codes: Array of latent codes [N, D]
        labels: Optional labels for coloring
        title: Title for plot
    
    Returns:
        figure: Matplotlib figure object
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    if latent_codes.shape[1] > 2:
        # Use t-SNE for dimensionality reduction
        if latent_codes.shape[0] > 1000:
            # Subsample for efficiency
            indices = np.random.choice(latent_codes.shape[0], 1000, replace=False)
            latent_codes = latent_codes[indices]
            if labels is not None:
                labels = labels[indices]
        
        # First reduce with PCA if very high dimensional
        if latent_codes.shape[1] > 50:
            pca = PCA(n_components=50)
            latent_codes = pca.fit_transform(latent_codes)
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        coords = tsne.fit_transform(latent_codes)
    else:
        coords = latent_codes
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                           c=labels, cmap='tab10', 
                           alpha=0.7, s=30)
        plt.colorbar(scatter, ax=ax, label='Class/Style')
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=30)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig


def visualize_latent_hierarchy(model, data_loader, device='cuda'):
    """
    Visualize the two-level latent space structure.
    
    TODO:
    1. Encode all data to get z_high and z_low
    2. Use t-SNE to visualize z_high (colored by genre)
    3. For each z_high cluster, show z_low variations
    4. Create hierarchical visualization
    """

    """
    層次潛空間視覺化：
      1) 編碼所有樣本，收集 z_high / z_low 與（可選）labels
      2) z_high 用 t-SNE → 2D 散點，大圖（顏色 = 標籤或自動分群）
      3) 挑幾個 z_high 群（或標籤），各自把其成員的 z_low 做降維 → 小圖
    回傳：
      figs: {'zhigh': fig1, 'zlow_panels': fig2}（可能為 None 若沒有樣本）
      data: dict，含 z_high、z_low、labels、cluster_id、tsne/pca 結果
    """
    model.eval().to(device)

    # --------- 1) 收集 latent -----------
    z_high_list, z_low_list, label_list = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                x = batch[0]
                y = batch[1] if len(batch) > 1 else None
            else:
                x, y = batch, None

            x = x.to(device).float()  # [B,16,9]
            mu_l, lv_l, mu_h, lv_h = model.encode_hierarchy(x)


            z_high_list.append(mu_h.detach().cpu())
            z_low_list.append(mu_l.detach().cpu())

            B = x.size(0)
            if y is None:
                label_list.append(torch.full((B,), -1))
            else:
                label_list.append(y.detach().cpu() if torch.is_tensor(y) else torch.tensor(y))

    if len(z_high_list) == 0:
        print("visualize_latent_hierarchy: No data to visualize latent hierarchy")
        return 



    z_high = torch.cat(z_high_list, dim=0).numpy()  # [N, Dh]
    z_low  = torch.cat(z_low_list,  dim=0).numpy()  # [N, Dl]
    labels = torch.cat(label_list,  dim=0).numpy()  # [N]


    Path("results/latent_analysis/").mkdir(parents=True, exist_ok=True)
    fig_high = plot_latent_space_2d(z_high, labels=(labels if (labels >= 0).any() else None))
    fig_high.savefig(f"results/latent_analysis/t-SNE_visualization_z_high.png", dpi=200)
    plt.close(fig_high)


    # --------- 3) 每個 z_high 群裡的 z_low 分布（小圖面板）-----------
    # 先決定要顯示哪些群（以群大小排序，取前 zlow_plot_topk 個）

    if (labels >= 0).any():
        for c in np.unique(labels):
            idx = (labels == c)
            z_low_c = z_low[idx]
            if z_low_c.shape[0] < 2:
                continue

            fig_low = plot_latent_space_2d(z_low_c)
            fig_low.savefig(f"results/latent_analysis/t-SNE_visualization_z_low_{int(c)}.png", dpi=200)
            plt.close(fig_low)



def interpolate_styles(model, pattern1, pattern2, n_steps=10, device='cuda', temperature=1.0, threshold=0.5):
    """
    Interpolate between two drum patterns at both latent levels.
    
    TODO:
    1. Encode both patterns to get latents
    2. Interpolate z_high (style transition)
    3. Interpolate z_low (variation transition)
    4. Decode and visualize both paths
    5. Compare smooth vs abrupt transitions
    """

    """
    Interpolate in latent space between two drum patterns.
    Returns prob grids for:
      - style_path:   z_high interpolated, z_low fixed at pattern1
      - variation_path: z_low interpolated, z_high fixed at pattern1
      - both_path:    both z_high & z_low interpolated
      - (optional) abrupt baseline in data space
    """

    model.to(device).eval()

    def to_batch(x):
        if x.dim() == 2:     # [16,9]
            x = x.unsqueeze(0)
        return x.to(device).float()  # [1,16,9]

    x1 = to_batch(pattern1)
    x2 = to_batch(pattern2)

    # 取編碼參數
    mu_l1, lv_l1, mu_h1, lv_h1 = model.encode_hierarchy(x1)
    mu_l2, lv_l2, mu_h2, lv_h2 = model.encode_hierarchy(x2)

    # 用均值或抽樣作為端點 latent
    base_l1, base_h1, base_l2, base_h2 = mu_l1, mu_h1, mu_l2, mu_h2

    # 建插值係數
    alphas = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)  # [n,1]

    # z_high path（z_low 固定為 pattern1 的）
    z_high_path = (1 - alphas) * base_h1 + alphas * base_h2   # [n,Dh]
    z_low_path   = (1 - alphas) * base_l1 + alphas * base_l2   # [n,Dl]

    z_low_fixed = base_l1.expand_as(z_low_path)               # [n,Dl]
    logits_style = model.decode_hierarchy(z_high_path, z_low=z_low_fixed, temperature=temperature)
    probs_style  = torch.sigmoid(logits_style)

    # z_low path（z_high 固定為 pattern1 的）
    # z_low_path   = (1 - alphas) * base_l1 + alphas * base_l2   # [n,Dl]
    z_high_fixed = base_h1.expand_as(z_high_path)                # [n,Dh]
    logits_vari  = model.decode_hierarchy(z_high_fixed, z_low=z_low_path, temperature=temperature)
    probs_vari   = torch.sigmoid(logits_vari)

    # both path（兩者都插）
    logits_both  = model.decode_hierarchy(z_high_path, z_low=z_low_path, temperature=temperature)
    probs_both   = torch.sigmoid(logits_both)

    #
    probs_style_bin = (probs_style > threshold).float()
    probs_vari_bin = (probs_vari > threshold).float()
    probs_both_bin = (probs_both > threshold).float()

    Path("results/generated_patterns/").mkdir(parents=True, exist_ok=True)
    for i in range(n_steps):
        fig = plot_drum_pattern(probs_style_bin[i])
        fig.savefig(f"results/generated_patterns/interpolation_sequences_style_{i}.png", dpi=200)
        plt.close(fig)

        fig1 = plot_drum_pattern(probs_vari_bin[i])
        fig1.savefig(f"results/generated_patterns/interpolation_sequences_variation_{i}.png", dpi=200)
        plt.close(fig1)

    out = {
        'style_path':     probs_style.cpu(),     # [n,16,9]
        'variation_path': probs_vari.cpu(),      # [n,16,9]
        'both_path':      probs_both.cpu(),      # [n,16,9]
    }

    return out


def style_transfer_example(model, pattern1, pattern2, pattern1_style, pattern2_style, device='cuda', threshold = 0.5):
    model.eval().to(device)
    
    def to_batch(x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x.to(device).float()

    xA = to_batch(pattern1)
    xB = to_batch(pattern2)

    mu_lA, logvar_lA, mu_hA, logvar_hA = model.encode_hierarchy(xA)
    mu_lB, logvar_lB, mu_hB, logvar_hB = model.encode_hierarchy(xB)

    # 用 mu 當 latent（比較穩定）
    z_low_A, z_high_A = mu_lA, mu_hA
    z_low_B, z_high_B = mu_lB, mu_hB

    # A 的細節 + B 的風格
    logits_A2B = model.decode_hierarchy(z_high_B, z_low=z_low_A)
    # B 的細節 + A 的風格
    logits_B2A = model.decode_hierarchy(z_high_A, z_low=z_low_B)

    probs_A2B = torch.sigmoid(logits_A2B).squeeze(0).cpu() # [16, 9]
    probs_B2A = torch.sigmoid(logits_B2A).squeeze(0).cpu() # [16, 9]
    probs_A2B_bin = (probs_A2B > threshold).float()
    probs_B2A_bin = (probs_B2A > threshold).float()

    Path("results/generated_patterns/").mkdir(parents=True, exist_ok=True)
    fig = plot_drum_pattern(probs_A2B_bin)
    fig.savefig(f"results/generated_patterns/style_transfer_{pattern1_style}_to_{pattern2_style}.png", dpi=200)
    plt.close(fig)

    fig1 = plot_drum_pattern(probs_B2A_bin)
    fig1.savefig(f"results/generated_patterns/style_transfer_{pattern2_style}_to_{pattern1_style}.png", dpi=200)
    plt.close(fig1)

    return probs_A2B, probs_B2A


def measure_disentanglement(model, data_loader, device='cuda'):
    """
    Measure how well the hierarchy disentangles style from variation.
    
    TODO:
    1. Group patterns by genre
    2. Compute z_high variance within vs across genres
    3. Compute z_low variance for same genre
    4. Return disentanglement metrics
    """
    
    model.eval().to(device)
    z_high_list, z_low_list, label_list = [], [], []

    # 1) 收集 z_high / z_low 與標籤
    for batch in data_loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            # 沒標籤就沒法做這個指標
            raise ValueError("measure_disentanglement: need data (x, style_label)")

        x = x.to(device).float()
        mu_low, logvar_low, mu_high, logvar_high = model.encode_hierarchy(x)
        z_high_list.append(mu_high.detach().cpu().numpy())
        z_low_list.append(mu_low.detach().cpu().numpy())
        label_list.append(y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y))

    z_high_list = np.concatenate(z_high_list, axis=0)  # [N, Dh]
    z_low_list = np.concatenate(z_low_list, axis=0)  # [N, Dl]
    label_list = np.concatenate(label_list,  axis=0).astype(int)  # [N]
    
    N  = z_high_list.shape[0]
    classes = np.unique(label_list)
    K = len(classes)

    # 2) z_high: within / between variance
    # 全體均值
    mu_all = z_high_list.mean(axis=0, keepdims=True)  # [1, Dh]
    # 逐類均值與大小
    mu_c = {}
    n_c  = {}
    for c in classes:
        idx = (label_list == c)
        mu_c[c] = z_high_list[idx].mean(axis=0, keepdims=True)
        n_c[c]  = idx.sum()

    # within-class variance（類內）
    within_terms = []
    for c in classes:
        idx = (label_list == c)
        dif = z_high_list[idx] - mu_c[c]           # [n_c, Dh]
        within_terms.append((dif ** 2).sum())
    high_within_var = float(np.sum(within_terms) / max(1, N - K))

    # between-class variance（類間）
    between_terms = []
    for c in classes:
        dif = (mu_c[c] - mu_all)          # [1, Dh]
        between_terms.append(n_c[c] * (dif ** 2).sum())
    high_between_var = float(np.sum(between_terms) / max(1, K - 1))

    high_separation = float(high_between_var / (high_within_var + 1e-8))

    # 3) z_low: 同風格內的變異（希望有多樣性）
    low_within_terms = []
    for c in classes:
        idx = (label_list == c)
        if idx.sum() <= 1:
            continue
        mu_l_c = z_low_list[idx].mean(axis=0, keepdims=True)
        dif = z_low_list[idx] - mu_l_c
        low_within_terms.append((dif ** 2).sum())
    low_within_var = float(np.sum(low_within_terms) / max(1, N - K))

    # 4) 線性探測（train/test 簡易劃分）
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    split = int(0.7 * N)
    tr, te = perm[:split], perm[split:]

    # z_high 線性可分性
    clf_h = LogisticRegression(max_iter=200, multi_class='auto')
    clf_h.fit(z_high_list[tr], label_list[tr])
    acc_high = float(accuracy_score(label_list[te], clf_h.predict(z_high_list[te])))

    # z_low 線性可分性（理想：接近隨機）
    clf_l = LogisticRegression(max_iter=200, multi_class='auto')
    clf_l.fit(z_low_list[tr], label_list[tr])
    acc_low = float(accuracy_score(label_list[te], clf_l.predict(z_low_list[te])))

    results = {
        "high_within_var": high_within_var,
        "high_between_var": high_between_var,
        "high_separation": high_separation,
        "low_within_var": low_within_var,
        "acc_high_linear_probe": acc_high,
        "acc_low_linear_probe": acc_low,
        "n_classes": int(K),
        "n_samples": int(N),
    }

    Path("results/latent_analysis/").mkdir(parents=True, exist_ok=True)
    with open("results/latent_analysis/disentanglement_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def drum_pattern_validity(pattern):
    """
    Check if a drum pattern is musically valid.
    
    Args:
        pattern: Binary tensor [16, 9] or [batch, 16, 9]
    
    Returns:
        validity_score: Float between 0 and 1
    """
    if pattern.dim() == 3:
        # Batch mode
        scores = []
        for i in range(pattern.shape[0]):
            scores.append(drum_pattern_validity(pattern[i]))
        return np.mean(scores)
    
    pattern = pattern.cpu().numpy()
    
    # Check basic musical constraints
    score = 1.0
    
    # 1. Pattern should not be empty
    if pattern.sum() == 0:
        return 0.0
    
    # 2. Pattern should not be too dense (> 50% filled)
    density = pattern.sum() / (16 * 9)
    if density > 0.5:
        score *= 0.8
    
    # 3. Should have some kick drum (instrument 0)
    if pattern[:, 0].sum() == 0:
        score *= 0.7
    
    # 4. Should have some rhythmic structure (not random)
    # Check for repeating patterns
    first_half = pattern[:8]
    second_half = pattern[8:]
    similarity = np.sum(first_half == second_half) / (8 * 9)
    
    if similarity < 0.3:  # Too random
        score *= 0.8
    
    return score

def check_musically_valid(model, data_loader, device='cuda', threshold= 0.5):

    model = model.to(device).eval()

    scores = []
    

    for batch in data_loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x = batch[0]
        else:
            x = batch

        x = x.to(device).float()

        recon, mu_low, logvar_low, mu_high, logvar_high = model(x, beta=1.0, temperature=1.0)

        probs = torch.sigmoid(recon)
        probs_bin = (probs > threshold).float()

        for i in range(probs_bin.size(0)):
            score = drum_pattern_validity(probs_bin[i])

            scores.append(score)

    return float(np.mean(scores)) if len(scores) > 0 else 0.0



def controllable_generation(model, genre_labels, device='cuda'):
    """
    Test controllable generation using the hierarchy.
    
    TODO:
    1. Learn genre embeddings in z_high space
    2. Generate patterns with specified genre
    3. Control complexity via z_low sampling temperature
    4. Evaluate genre classification accuracy
    """
    pass