"""
Hierarchical VAE for drum pattern generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalDrumVAE(nn.Module):
    def __init__(self, z_high_dim=4, z_low_dim=12):
        """
        Two-level VAE for drum patterns.
        
        The architecture uses a hierarchy of latent variables where z_high
        encodes style/genre information and z_low encodes pattern variations.
        
        Args:
            z_high_dim: Dimension of high-level latent (style)
            z_low_dim: Dimension of low-level latent (variation)
        """
        super().__init__()
        self.z_high_dim = z_high_dim
        self.z_low_dim = z_low_dim
        
        # Encoder: pattern → z_low → z_high
        # We use 1D convolutions treating the pattern as a sequence
        self.encoder_low = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=3, padding=1),  # [16, 9] → [16, 32]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # → [8, 64]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # → [4, 128]
            nn.ReLU(),
            nn.Flatten()  # → [512]
        )
        
        # Low-level latent parameters
        self.fc_mu_low = nn.Linear(512, z_low_dim)
        self.fc_logvar_low = nn.Linear(512, z_low_dim)
        
        # Encoder from z_low to z_high
        self.encoder_high = nn.Sequential(
            nn.Linear(z_low_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # High-level latent parameters
        self.fc_mu_high = nn.Linear(32, z_high_dim)
        self.fc_logvar_high = nn.Linear(32, z_high_dim)
        
        # Decoder: z_high → z_low → pattern
        # TODO: Implement decoder architecture
        # Mirror the encoder structure
        # Use transposed convolutions for upsampling

        # ---- Conditional prior: p(z_low | z_high) ----
        # 輸出拼成 [mu_p_low, logvar_p_low]，供抽樣與 KL 使用
        self.prior_low = nn.Sequential(
            nn.Linear(z_high_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * z_low_dim)
        )

        # ---- Decoder: [z_high ; z_low] → pattern logits ----
        # 先用 Linear 拉成卷積張量，之後用 ConvTranspose1d 上採樣回 16 steps
        self.decoder_pre = nn.Sequential(
            nn.Linear(z_high_dim + z_low_dim, 128 * 4),
            nn.ReLU()
        )

        self.decoder_deconv = nn.Sequential(
            # [B, 128, 4] → [B, 64, 8]
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # [B, 64, 8] → [B, 32, 16]
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 映射到 9 個聲部通道
            nn.Conv1d(32, 9, kernel_size=3, padding=1)  # 最終輸出 logits（未 sigmoid）
        )
        
    def encode_hierarchy(self, x):
        """
        Encode pattern to both latent levels.
        
        Args:
            x: Drum patterns [batch_size, 16, 9]
            
        Returns:
            mu_low, logvar_low: Parameters for q(z_low|x)
            mu_high, logvar_high: Parameters for q(z_high|z_low)
        """
        # Reshape for Conv1d: [batch, 16, 9] → [batch, 9, 16]
        x = x.transpose(1, 2).float()
        
        # TODO: Encode to z_low parameters
        # TODO: Sample z_low using reparameterization
        # TODO: Encode z_low to z_high parameters
        
        # 2) 編碼到低層潛變數參數 (mu_low, logvar_low)
        h = self.encoder_low(x)                 # [B, 512]
        mu_low = self.fc_mu_low(h)              # [B, z_low_dim]
        logvar_low = self.fc_logvar_low(h)      # [B, z_low_dim]

        # 3) 抽樣 z_low（重參數化）
        z_low = self.reparameterize(mu_low, logvar_low) # [B, z_low_dim]

        # 4) 由 z_low 編碼到高層潛變數參數 (mu_high, logvar_high)
        h_high = self.encoder_high(z_low)           # [B, 32]
        mu_high = self.fc_mu_high(h_high)           # [B, z_high_dim]
        logvar_high = self.fc_logvar_high(h_high)   # [B, z_high_dim]
        
        return mu_low, logvar_low, mu_high, logvar_high
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling.
        
        TODO: Implement
        z = mu + eps * std where eps ~ N(0,1)
        """
        
        # logvar = torch.clamp(logvar, min=-10.0, max=10.0) # 可選：數值穩定，避免 exp 溢位或過小
        std = torch.exp(0.5 * logvar)   # 把 logσ² 轉成 σ
        eps = torch.randn_like(std)     # 產生 N(0,1) 噪音
        z = mu + eps * std              # 公式: z = μ + σ·ε
        
        return z

    
    def decode_hierarchy(self, z_high, z_low=None, temperature=1.0):
        """
        Decode from latent variables to pattern.
        
        Args:
            z_high: High-level latent code
            z_low: Low-level latent code (if None, sample from prior)
            temperature: Temperature for binary output (lower = sharper)
            
        Returns:
            pattern_logits: Logits for binary pattern [batch, 16, 9]
        """
        # TODO: If z_low is None, sample from conditional prior p(z_low|z_high)
        # TODO: Decode z_high and z_low to pattern logits
        # TODO: Apply temperature scaling before sigmoid
        
        B = z_high.size(0)

        # 1) 條件先驗 p(z_low | z_high)
        prior_out = self.prior_low(z_high)                 # [B, 2*z_low_dim]
        mu_p_low, logvar_p_low = torch.chunk(prior_out, 2, dim=-1)

        # 2) 若未給 z_low，從條件先驗抽樣（純生成或評估 prior-sample 用）
        if z_low is None:
            z_low = self.reparameterize(mu_p_low, logvar_p_low)  # [B, z_low_dim]

        # 3) 串接兩層 latent，進解碼器
        z = torch.cat([z_high, z_low], dim=-1)             # [B, z_high_dim + z_low_dim]
        h = self.decoder_pre(z)                            # [B, 128*4]
        h = h.view(B, 128, 4)                              # [B, 128, 4]

        y = self.decoder_deconv(h)                         # [B, 9, 16]
        pattern_logits = y.transpose(1, 2)                 # → [B, 16, 9]

        # 4) 溫度縮放（通常只在推論輸出機率/取樣時使用）
        if temperature != 1.0:
            pattern_logits = pattern_logits / temperature

        return pattern_logits
    
    
    def forward(self, x, beta=1.0, temperature=1.0):
        """
        Full forward pass with loss computation.
        
        Args:
            x: Input patterns [batch_size, 16, 9]
            beta: KL weight for beta-VAE (use < 1 to prevent collapse)
            
        Returns:
            recon: Reconstructed patterns
            mu_low, logvar_low, mu_high, logvar_high: Latent parameters
        """
        # TODO: Encode, decode, compute losses
        
        # 只做 encode→sample→decode；把 logits 交給外部算 loss
        mu_low, logvar_low, mu_high, logvar_high = self.encode_hierarchy(x)
        z_low  = self.reparameterize(mu_low,  logvar_low)
        z_high = self.reparameterize(mu_high, logvar_high)

        # 建議 decode 回傳 logits 與先驗參數，方便外部算 KL
        logits = self.decode_hierarchy(z_high, z_low=z_low, temperature=temperature)

        return logits, mu_low, logvar_low, mu_high, logvar_high