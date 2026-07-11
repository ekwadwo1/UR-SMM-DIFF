#!/usr/bin/env python3
"""
models/vqgan3d.py — 3D VQGAN for Latent Space Compression  (Phase 4-A)
=======================================================================

Paper Eqs. 2-3 (Section 3, Methodology):
    z0 = E(x0)   (Eq. 2)       x_hat_0 = D(z_hat_0)   (Eq. 3)

CRITICAL: Diffusion operates on the CONTINUOUS pre-quantization latent
z0 = E(x0), NOT the post-VQ quantized z_q.

Components: Encoder3D, Decoder3D, VectorQuantizer3D, VQGAN3D,
            Discriminator3D, VQGANLoss, ssim3d, compute_psnr

Hardware: 2x NVIDIA RTX 5880 Ada 48 GB, CUDA 11.8
"""
from __future__ import annotations
import math, sys
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== Building Blocks =====================

class GroupNormSiLU(nn.Module):
    """GroupNorm(32) + SiLU."""
    def __init__(self, ch: int) -> None:
        super().__init__()
        ng = min(32, ch)
        while ch % ng != 0 and ng > 1:
            ng -= 1
        self.norm = nn.GroupNorm(ng, ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(x))

class ResBlock3D(nn.Module):
    """Conv3d(3^3)+GN+SiLU+Conv3d(3^3)+skip. Paper Section 3 backbone block."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm1 = GroupNormSiLU(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = GroupNormSiLU(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.norm1(x))
        h = self.conv2(self.norm2(h))
        return h + self.skip(x)

class Downsample3D(nn.Module):
    """Strided conv for x2 spatial downsampling (not pooling)."""
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample3D(nn.Module):
    """Trilinear interp + Conv3d for x2 spatial upsampling."""
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="trilinear", align_corners=False)
        return self.conv(x)

# ===================== Encoder3D (Eq. 2) =====================

class Encoder3D(nn.Module):
    """
    3D convolutional encoder -- Eq. (2): z0 = E(x0).
    n_res_blocks ResBlock3D per stage, strided-conv downsampling.
    r=4: 2 downsamples (128->64->32).  r=8: 3 downsamples (128->64->32->16).
    """
    def __init__(self, in_channels: int = 4, latent_dim: int = 4,
                 base_channels: int = 64, channel_mult: Tuple[int,...] = (1,2,4,8),
                 n_res_blocks: int = 2, downsample_factor: int = 4) -> None:
        super().__init__()
        n_down = int(math.log2(downsample_factor))
        assert 2**n_down == downsample_factor, "r must be power of 2"
        n_stages = n_down + 1
        assert n_stages <= len(channel_mult)
        self.n_down = n_down
        self.n_res_blocks = n_res_blocks
        chs = [base_channels * m for m in channel_mult[:n_stages]]

        self.in_conv = nn.Conv3d(in_channels, chs[0], 3, padding=1)
        blocks, downs = [], []
        ch_prev = chs[0]
        for i in range(n_stages):
            ch = chs[i]
            for _ in range(n_res_blocks):
                blocks.append(ResBlock3D(ch_prev, ch)); ch_prev = ch
            if i < n_down:
                downs.append(Downsample3D(ch))
        self.blocks = nn.ModuleList(blocks)
        self.downs = nn.ModuleList(downs)
        self.out_norm = GroupNormSiLU(chs[-1])
        self.out_conv = nn.Conv3d(chs[-1], latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x:[B,C,H,W,D] -> z:[B,c,H/r,W/r,D/r]"""
        h = self.in_conv(x)
        idx = 0
        for i in range(self.n_down + 1):
            for _ in range(self.n_res_blocks):
                h = self.blocks[idx](h); idx += 1
            if i < self.n_down:
                h = self.downs[i](h)
        return self.out_conv(self.out_norm(h))

# ===================== Decoder3D (Eq. 3) =====================

class Decoder3D(nn.Module):
    """
    3D convolutional decoder -- Eq. (3): x_hat_0 = D(z_hat_0).
    Mirror of Encoder3D with trilinear upsample + conv. No output activation.
    """
    def __init__(self, out_channels: int = 4, latent_dim: int = 4,
                 base_channels: int = 64, channel_mult: Tuple[int,...] = (1,2,4,8),
                 n_res_blocks: int = 2, downsample_factor: int = 4) -> None:
        super().__init__()
        n_up = int(math.log2(downsample_factor))
        n_stages = n_up + 1
        chs = [base_channels * m for m in channel_mult[:n_stages]]
        chs_rev = list(reversed(chs))
        self.n_up = n_up
        self.n_res_blocks = n_res_blocks

        self.in_conv = nn.Conv3d(latent_dim, chs_rev[0], 3, padding=1)
        blocks, ups = [], []
        ch_prev = chs_rev[0]
        for i in range(n_stages):
            ch = chs_rev[i]
            for _ in range(n_res_blocks):
                blocks.append(ResBlock3D(ch_prev, ch)); ch_prev = ch
            if i < n_up:
                ups.append(Upsample3D(ch))
        self.blocks = nn.ModuleList(blocks)
        self.ups = nn.ModuleList(ups)
        self.out_norm = GroupNormSiLU(chs_rev[-1])
        self.out_conv = nn.Conv3d(chs_rev[-1], out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z:[B,c,h,w,d] -> x_recon:[B,C,H,W,D]"""
        h = self.in_conv(z)
        idx = 0
        for i in range(self.n_up + 1):
            for _ in range(self.n_res_blocks):
                h = self.blocks[idx](h); idx += 1
            if i < self.n_up:
                h = self.ups[i](h)
        return self.out_conv(self.out_norm(h))

# ===================== VectorQuantizer3D =====================

class VectorQuantizer3D(nn.Module):
    """
    VQ with EMA codebook + straight-through estimator.
    Used ONLY during VQGAN pre-training. Diffusion uses continuous z0.
    """
    def __init__(self, n_embed: int = 8192, embed_dim: int = 4,
                 commitment_cost: float = 0.25, ema_decay: float = 0.99) -> None:
        super().__init__()
        self.n_embed = n_embed; self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost; self.ema_decay = ema_decay
        self.register_buffer("embedding", torch.randn(n_embed, embed_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(n_embed))
        self.register_buffer("ema_embed_sum", self.embedding.clone())
        self._init = False

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z:[B,c,h,w,d] -> z_q, vq_loss, perplexity. Straight-through grad."""
        B,C,H,W,D = z.shape
        flat = z.permute(0,2,3,4,1).reshape(-1, C)
        if self.training and not self._init:
            n = min(flat.shape[0], self.n_embed)
            self.embedding.data[:n] = flat[torch.randperm(flat.shape[0])[:n]].detach()
            self._init = True
        d = flat.pow(2).sum(1,keepdim=True) + self.embedding.pow(2).sum(1) - 2*flat@self.embedding.t()
        idx = d.argmin(1)
        zq = self.embedding[idx]
        if self.training:
            oh = F.one_hot(idx, self.n_embed).float()
            self.ema_cluster_size.mul_(self.ema_decay).add_(oh.sum(0), alpha=1-self.ema_decay)
            self.ema_embed_sum.mul_(self.ema_decay).add_(oh.t()@flat.detach(), alpha=1-self.ema_decay)
            n = self.ema_cluster_size.sum()
            sm = (self.ema_cluster_size+1e-5)/(n+self.n_embed*1e-5)*n
            self.embedding.data.copy_(self.ema_embed_sum / sm.unsqueeze(1))
        vq_loss = self.commitment_cost * F.mse_loss(flat, zq.detach())
        zq = flat + (zq - flat).detach()
        zq = zq.reshape(B,H,W,D,C).permute(0,4,1,2,3)
        avg = F.one_hot(idx, self.n_embed).float().mean(0)
        perp = torch.exp(-torch.sum(avg * torch.log(avg + 1e-10)))
        return zq, vq_loss, perp

# ===================== VQGAN3D (Eqs. 2-3) =====================

class VQGAN3D(nn.Module):
    """
    3D VQGAN -- Eqs. (2)-(3). Three forward modes:
      'vqgan_train': Encoder -> VQ -> Decoder (VQGAN pre-training)
      'encode':      Encoder only -> continuous z0 (for diffusion)
      'decode':      Decoder only -> x_hat_0 (for diffusion)
    """
    def __init__(self, in_channels: int = 4, latent_dim: int = 4,
                 base_channels: int = 64, channel_mult: Tuple[int,...] = (1,2,4,8),
                 n_res_blocks: int = 2, downsample_factor: int = 4,
                 n_embed: int = 8192, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.downsample_factor = downsample_factor
        self.latent_dim = latent_dim
        self.encoder = Encoder3D(in_channels, latent_dim, base_channels,
                                  channel_mult, n_res_blocks, downsample_factor)
        self.decoder = Decoder3D(in_channels, latent_dim, base_channels,
                                  channel_mult, n_res_blocks, downsample_factor)
        self.quantizer = VectorQuantizer3D(n_embed, latent_dim, commitment_cost)

    def forward(self, x: torch.Tensor, mode: str = "vqgan_train"):
        if mode == "vqgan_train":
            z_pre = self.encoder(x)
            z_q, vq_loss, perp = self.quantizer(z_pre)
            x_recon = self.decoder(z_q)
            return x_recon, z_q, vq_loss, perp
        elif mode == "encode":
            return self.encoder(x)          # continuous z0 -- NO quantization
        elif mode == "decode":
            return self.decoder(x)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, mode="encode")
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z, mode="decode")

# ===================== Discriminator3D =====================

class Discriminator3D(nn.Module):
    """3D PatchGAN with spectral norm. Returns (logits, features)."""
    def __init__(self, in_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        ndf = base_channels
        def _b(ic,oc,s=2):
            return nn.Sequential(nn.utils.spectral_norm(nn.Conv3d(ic,oc,4,stride=s,padding=1)),
                                 nn.LeakyReLU(0.2, inplace=True))
        self.blocks = nn.ModuleList([_b(in_channels,ndf,2), _b(ndf,ndf*2,2),
                                      _b(ndf*2,ndf*4,2), _b(ndf*4,ndf*4,1)])
        self.final = nn.utils.spectral_norm(nn.Conv3d(ndf*4, 1, 4, stride=1, padding=1))
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats = []
        h = x
        for blk in self.blocks:
            h = blk(h); feats.append(h)
        return self.final(h), feats

# ===================== SSIM3D =====================

def ssim3d(x: torch.Tensor, y: torch.Tensor, window_size: int = 7,
           data_range: float = 1.0) -> torch.Tensor:
    """3D Structural Similarity (separable Gaussian kernel)."""
    C1 = (0.01*data_range)**2; C2 = (0.03*data_range)**2
    dev, ch = x.device, x.shape[1]
    g = torch.arange(window_size, dtype=torch.float32, device=dev) - window_size//2
    g = torch.exp(-g**2 / (2*1.5**2)); g = g / g.sum()
    k = (g[:,None,None]*g[None,:,None]*g[None,None,:]).unsqueeze(0).unsqueeze(0).expand(ch,-1,-1,-1,-1)
    p = window_size // 2
    mx = F.conv3d(x,k,padding=p,groups=ch); my = F.conv3d(y,k,padding=p,groups=ch)
    sxx = F.conv3d(x*x,k,padding=p,groups=ch)-mx**2
    syy = F.conv3d(y*y,k,padding=p,groups=ch)-my**2
    sxy = F.conv3d(x*y,k,padding=p,groups=ch)-mx*my
    return ((2*mx*my+C1)*(2*sxy+C2)/((mx**2+my**2+C1)*(sxx.clamp(0)+syy.clamp(0)+C2))).mean()

# ===================== VQGANLoss =====================

class VQGANLoss(nn.Module):
    """L = L_recon(L1+0.1*SSIM) + lam_vq*L_vq + lam_perc*L_feat + lam_adv*L_adv."""
    def __init__(self, lambda_vq: float = 1.0, lambda_perc: float = 0.1,
                 lambda_adv: float = 0.1, disc_start_step: int = 2000) -> None:
        super().__init__()
        self.lambda_vq=lambda_vq; self.lambda_perc=lambda_perc
        self.lambda_adv=lambda_adv; self.disc_start_step=disc_start_step

    def generator_loss(self, x, x_rec, vq_loss, disc_fake_logits=None,
                       disc_fake_feats=None, disc_real_feats=None, global_step=0):
        l1 = F.l1_loss(x_rec, x)
        dr = (x.max()-x.min()).clamp(min=1e-6).item()
        sv = ssim3d(x, x_rec, data_range=dr)
        lr = l1 + 0.1*(1.0-sv)
        lt = lr + self.lambda_vq * vq_loss
        logs = {"l1":l1.item(),"ssim":sv.item(),"l_recon":lr.item(),"vq_loss":vq_loss.item()}
        if global_step >= self.disc_start_step and disc_fake_logits is not None:
            la = -disc_fake_logits.mean()
            lf = torch.tensor(0.0, device=x.device)
            if disc_fake_feats and disc_real_feats:
                for ff,rf in zip(disc_fake_feats,disc_real_feats):
                    lf = lf + F.l1_loss(ff, rf.detach())
                lf = lf / len(disc_fake_feats)
            lt = lt + self.lambda_adv*la + self.lambda_perc*lf
            logs.update({"l_adv_g":la.item(),"l_feat":lf.item()})
        logs["l_total_g"] = lt.item()
        return lt, logs

    def discriminator_loss(self, real_logits, fake_logits):
        lr = F.relu(1.0-real_logits).mean()
        lf = F.relu(1.0+fake_logits).mean()
        ld = 0.5*(lr+lf)
        return ld, {"l_disc":ld.item()}

# ===================== PSNR =====================

@torch.no_grad()
def compute_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x,y).item()
    if mse < 1e-12: return float("inf")
    dr = x.max().item() - x.min().item()
    return 10.0*math.log10(dr**2/mse) if dr > 1e-12 else 0.0

# ===================== Tests (Jupyter-safe) =====================

def _run_tests() -> None:
    """Shape + gradient tests for r=4 and r=8. No argparse."""
    print("="*70)
    print("  VQGAN3D -- Shape & Functionality Tests (Eqs. 2-3)")
    print("="*70)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for r in [4, 8]:
        print(f"\n{'─'*60}\n  r = {r}   device = {device}\n{'─'*60}")
        bc = 32 if device == "cpu" else 64
        nr = 1 if device == "cpu" else 2
        model = VQGAN3D(4,4,bc,(1,2,4,8),nr,r).to(device)
        x = torch.randn(1,4,128,128,128, device=device)
        h = 128 // r
        # encode
        z = model(x, mode="encode")
        assert z.shape == (1,4,h,h,h), f"encode fail: {z.shape}"
        print(f"  encode: [1,4,128^3] -> {list(z.shape)}  ✓")
        # decode
        xd = model(z, mode="decode")
        assert xd.shape == x.shape, f"decode fail: {xd.shape}"
        print(f"  decode: {list(z.shape)} -> {list(xd.shape)}  ✓")
        # vqgan_train
        xr, zq, vl, pp = model(x, mode="vqgan_train")
        assert xr.shape == x.shape and zq.shape == z.shape
        print(f"  train:  recon={list(xr.shape)} vq_loss={vl.item():.4f} perp={pp.item():.0f}  ✓")
        # gradient
        zt = model.encoder(x); zt.requires_grad_(True)
        zqt,_,_ = model.quantizer(zt); zqt.sum().backward()
        assert zt.grad is not None, "VQ grad broken!"
        print(f"  VQ straight-through gradient: ✓")
        # discriminator
        disc = Discriminator3D(4, bc//4).to(device)
        lo, fe = disc(x)
        print(f"  Discriminator: logits={list(lo.shape)}  ✓")
        # loss
        cr = VQGANLoss()
        gl, _ = cr.generator_loss(x, xr, vl)
        dr2, _ = disc(x); df, _ = disc(xr.detach())
        dl, _ = cr.discriminator_loss(dr2, df)
        print(f"  Loss: g={gl.item():.4f} d={dl.item():.4f}  ✓")
        # ssim
        ss = ssim3d(x, x, data_range=(x.max()-x.min()).item())
        assert ss.item() > 0.99
        print(f"  SSIM(x,x) = {ss.item():.6f}  ✓")
        # psnr
        print(f"  PSNR = {compute_psnr(x, xr):.2f} dB")
        # params
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
        del model, disc, x, z, xd, xr
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"\n{'='*70}\n  ALL TESTS PASSED ✓\n{'='*70}")

if __name__ == "__main__":
    _run_tests()
