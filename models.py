# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp, Attention
from einops import rearrange
from typing import Tuple, List

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        xp = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_msa.unsqueeze(1) * xp
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def power_of_2_closest_to_root(x: int):
    i = 1
    while (i * i < x):
        i <<= 1
    return i

class QFormerBasedCompressor(nn.Module):
    def __init__(self, query_seqlen:int, model_dim:int):
        super().__init__()
        self.query = nn.Parameter(torch.randn([query_seqlen, model_dim]))
        wq = power_of_2_closest_to_root(query_seqlen)
        hq = query_seqlen // wq
        self.pool = nn.Sequential(nn.Conv2d(model_dim, model_dim, kernel_size=(3,3), padding=(1,1)), nn.AdaptiveAvgPool2d((hq, wq)))
        self.ffn = nn.ModuleList([
            Mlp(in_features=model_dim, hidden_features=model_dim*2, out_features=model_dim*2, act_layer=nn.SiLU, drop=0)
            for _ in range(1)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(model_dim, model_dim>>6, bias=True, add_bias_kv=True, batch_first=True) for _ in range(1)])
        self.norm1 = nn.ModuleList([nn.RMSNorm(model_dim) for _ in range(1)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(model_dim) for _ in range(1)])
        self.blend = nn.Linear(2 * model_dim, model_dim)
            
        
    def forward(self, x:torch.Tensor):
        B, L, D = x.shape
        query = self.query.expand((B, -1, -1))
        wl = power_of_2_closest_to_root(L)

        r = rearrange(x, "B (H W) C -> B C H W", W=wl)
        r = self.pool(r)
        r = rearrange(r, "B C H W -> B (H W) C")
        for ffn, attn, norm1, norm2 in zip(self.ffn, self.attn, self.norm1, self.norm2):
            x = query + norm1(attn.forward(query, x, x)[0])
            x, gate = torch.chunk(ffn(x), 2, dim=-1)
            x = x + norm2(x) * gate
        
        x = self.blend(torch.cat([r, x], dim=-1))
        return x


class QFormerBasedCompressorV2(nn.Module):
    """Content-adaptive compressor (V2). Same interface/output as V1, but *selection* is
    content-aware instead of fixed: (1) FiLM-conditioned queries adapt to each image's
    global content, (2) a per-input-token importance logit biases the cross-attention —
    a soft, differentiable analogue of routing's 'which tokens matter'. Both signals are
    zero-initialised, so at init V2 behaves like a plain cross-attn compressor (stable,
    ~V1) and only becomes content-adaptive as training finds it useful."""
    def __init__(self, query_seqlen:int, model_dim:int, num_layers:int=1):
        super().__init__()
        self.query_seqlen = query_seqlen
        self.base_query = nn.Parameter(torch.empty(query_seqlen, model_dim))
        nn.init.trunc_normal_(self.base_query, std=0.02)
        # (1) image-conditioned FiLM on the queries (D -> 2D; zero-init => identity at start)
        self.query_film = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU(),
                                        nn.Linear(model_dim, 2 * model_dim))
        # (2) per-input-token importance logit (zero-init => uniform attention at start)
        self.importance = nn.Linear(model_dim, 1)
        # structural avg-pool prior branch (same as V1)
        wq = power_of_2_closest_to_root(query_seqlen)
        hq = query_seqlen // wq
        self.pool = nn.Sequential(nn.Conv2d(model_dim, model_dim, kernel_size=(3,3), padding=(1,1)),
                                  nn.AdaptiveAvgPool2d((hq, wq)))
        # add_bias_kv=False so the importance attn_mask aligns with exactly L keys (no +1 bias token)
        self.attn = nn.ModuleList([nn.MultiheadAttention(model_dim, model_dim>>6, bias=True, add_bias_kv=False, batch_first=True) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([Mlp(in_features=model_dim, hidden_features=model_dim*2, out_features=model_dim*2, act_layer=nn.SiLU, drop=0) for _ in range(num_layers)])
        self.norm1 = nn.ModuleList([nn.RMSNorm(model_dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(model_dim) for _ in range(num_layers)])
        self.blend = nn.Linear(2 * model_dim, model_dim)
        self._zero_adaptive()

    def _zero_adaptive(self):
        # zero-init the content-adaptive heads so V2 starts == a plain cross-attn compressor.
        # Re-called by SiT.initialize_weights() because its global xavier apply() overwrites these.
        nn.init.zeros_(self.query_film[-1].weight); nn.init.zeros_(self.query_film[-1].bias)
        nn.init.zeros_(self.importance.weight); nn.init.zeros_(self.importance.bias)

    def forward(self, x:torch.Tensor):
        B, L, D = x.shape
        # (1) content-conditioned queries: q = base * (1 + scale(ctx)) + shift(ctx)
        ctx = x.mean(dim=1)                                          # (B, D) global summary
        scale, shift = torch.chunk(self.query_film(ctx), 2, dim=-1)  # (B, D) each
        q = self.base_query.unsqueeze(0) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # (B, Q, D)
        # (2) importance -> additive bias on attention logits (B*heads, Q, L)
        imp = self.importance(x).squeeze(-1)                        # (B, L) per-key logit
        nH = self.attn[0].num_heads
        attn_bias = imp[:, None, :].expand(B, self.query_seqlen, L).repeat_interleave(nH, dim=0)
        # structural pooled prior
        wl = power_of_2_closest_to_root(L)
        r = rearrange(x, "B (H W) C -> B C H W", W=wl)
        r = rearrange(self.pool(r), "B C H W -> B (H W) C")
        for attn, ffn, norm1, norm2 in zip(self.attn, self.ffn, self.norm1, self.norm2):
            a = attn.forward(q, x, x, attn_mask=attn_bias, need_weights=False)[0]
            q = q + norm1(a)
            h, gate = torch.chunk(ffn(q), 2, dim=-1)
            q = q + norm2(h) * gate
        return self.blend(torch.cat([r, q], dim=-1))


class QFormerBasedCompressorV3(nn.Module):
    """Hard top-k routing compressor (V3). Like V2, but the importance signal makes a
    DISCRETE selection: only the top-K most important input tokens are kept, and the
    queries compress over that subset (closest to TREAD-style routing). A straight-through
    estimator routes gradient to the importance head for the KEPT tokens so it learns what
    to select. The avg-pool prior still runs over ALL tokens, so the low-freq skip stays
    complete even for dropped tokens. keep_ratio controls how aggressive the routing is.

    Note: at init the importance head is zero, so top-k ties resolve to the first K tokens
    (a mild spatial bias) — transient; it self-corrects once importance trains. STE gives
    gradient only to kept tokens (inherent to hard top-k); use V2 if that's too unstable."""
    def __init__(self, query_seqlen:int, model_dim:int, keep_ratio:float=0.5, num_layers:int=1):
        super().__init__()
        self.query_seqlen = query_seqlen
        self.keep_ratio = keep_ratio
        self.base_query = nn.Parameter(torch.empty(query_seqlen, model_dim))
        nn.init.trunc_normal_(self.base_query, std=0.02)
        self.query_film = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU(),
                                        nn.Linear(model_dim, 2 * model_dim))
        self.importance = nn.Linear(model_dim, 1)
        wq = power_of_2_closest_to_root(query_seqlen)
        hq = query_seqlen // wq
        self.pool = nn.Sequential(nn.Conv2d(model_dim, model_dim, kernel_size=(3,3), padding=(1,1)),
                                  nn.AdaptiveAvgPool2d((hq, wq)))
        self.attn = nn.ModuleList([nn.MultiheadAttention(model_dim, model_dim>>6, bias=True, add_bias_kv=False, batch_first=True) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([Mlp(in_features=model_dim, hidden_features=model_dim*2, out_features=model_dim*2, act_layer=nn.SiLU, drop=0) for _ in range(num_layers)])
        self.norm1 = nn.ModuleList([nn.RMSNorm(model_dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(model_dim) for _ in range(num_layers)])
        self.blend = nn.Linear(2 * model_dim, model_dim)
        self._zero_adaptive()

    def _zero_adaptive(self):
        # zero-init the adaptive heads (re-called after SiT's global xavier apply()).
        nn.init.zeros_(self.query_film[-1].weight); nn.init.zeros_(self.query_film[-1].bias)
        nn.init.zeros_(self.importance.weight); nn.init.zeros_(self.importance.bias)

    def forward(self, x:torch.Tensor):
        B, L, D = x.shape
        # content-conditioned queries (same as V2)
        ctx = x.mean(dim=1)
        scale, shift = torch.chunk(self.query_film(ctx), 2, dim=-1)
        q = self.base_query.unsqueeze(0) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # (B, Q, D)
        # hard top-K selection over importance scores (K static for a fixed L -> compile-safe)
        s = self.importance(x).squeeze(-1)                          # (B, L)
        K = max(self.query_seqlen, int(round(L * self.keep_ratio)))
        topv, topi = torch.topk(s, K, dim=1)                        # (B, K)
        x_kept = torch.gather(x, 1, topi.unsqueeze(-1).expand(-1, -1, D))  # (B, K, D)
        # straight-through: forward multiplies by 1 (pure selection), backward routes grad
        # through the kept tokens' scores so importance learns what to keep.
        gate = torch.sigmoid(topv).unsqueeze(-1)                    # (B, K, 1)
        x_kept = x_kept * (1 + gate - gate.detach())
        # structural pooled prior over ALL tokens (skip stays complete)
        wl = power_of_2_closest_to_root(L)
        r = rearrange(x, "B (H W) C -> B C H W", W=wl)
        r = rearrange(self.pool(r), "B C H W -> B (H W) C")
        for attn, ffn, norm1, norm2 in zip(self.attn, self.ffn, self.norm1, self.norm2):
            a = attn.forward(q, x_kept, x_kept, need_weights=False)[0]
            q = q + norm1(a)
            h, gate2 = torch.chunk(ffn(q), 2, dim=-1)
            q = q + norm2(h) * gate2
        return self.blend(torch.cat([r, q], dim=-1))


class QFormerDecompressor(nn.Module):
    """[PE fix] Perceiver-IO positional decoder. Instead of padding the compressed tokens
    with a shared learned mask token and self-attending (position-blind: every recovered
    slot starts identical), we cross-attend a bank of *learned positional queries* into the
    compressed tokens. Each output token is anchored to its own spatial slot, so the
    decompressor can place recovered detail where it belongs -- E-MMDiT (arXiv:2510.27135)
    ablates this "Position Reinforcement on reconstructed tokens" as ~10% FID (24.78->22.42),
    and it matters even more here since the Q-Former compressor pools away all spatial layout."""
    def __init__(self, model_dim:int, max_seqlen:int=4096):
        super().__init__()
        NUM_LAYERS = 1
        self.pos_query = nn.Parameter(torch.empty(max_seqlen, model_dim))
        nn.init.trunc_normal_(self.pos_query, std=0.02)
        self.cross = nn.ModuleList([
            nn.MultiheadAttention(model_dim, model_dim>>6, bias=True, add_bias_kv=True, batch_first=True)
            for _ in range(NUM_LAYERS)])
        self.ffn = nn.ModuleList([
            Mlp(in_features=model_dim, hidden_features=model_dim*4, out_features=model_dim*2, act_layer=nn.SiLU, drop=0)
            for _ in range(NUM_LAYERS)])
        self.norm1 = nn.ModuleList([nn.RMSNorm(model_dim) for _ in range(NUM_LAYERS)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(model_dim) for _ in range(NUM_LAYERS)])


    def forward(self, x:torch.Tensor, original_seqlen:int):
        B = x.shape[0]
        assert original_seqlen <= self.pos_query.shape[0], \
            f"decompressor pos_query bank ({self.pos_query.shape[0]}) too small for seqlen {original_seqlen}"
        q = self.pos_query[:original_seqlen].expand(B, -1, -1)   # (B, L, D) position queries
        for cross, ffn, norm1, norm2 in zip(self.cross, self.ffn, self.norm1, self.norm2):
            q = q + norm1(cross(q, x, x)[0])                     # cross-attend queries -> compressed tokens
            q, gate = torch.chunk(ffn(q), 2, dim=-1)
            q = q + norm2(q) * gate
        return q
    
class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        query_seqlen:int=16,
        aux_recon:bool=False,
        aux_recon_weight:float=0.1,
        compressor_version:int=1,   # 1 = fixed pool+xattn (default); 2 = soft content-adaptive; 3 = hard top-k routing
        keep_ratio:float=0.5,       # V3 only: fraction of input tokens kept by the top-k router
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        # [aux #1] bottleneck-autoencoder loss: the compressed tokens must reconstruct
        # the pre-compression features, pressuring the compressor to preserve
        # reconstructable detail so token reduction can be pushed past 75%.
        self.aux_recon = aux_recon
        self.aux_recon_weight = aux_recon_weight
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        self.layer_landmarks = [depth // 6, depth - (depth // 6)]

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.detailer = nn.Linear(hidden_size + in_channels, hidden_size)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        if compressor_version == 3:
            self.compressor = QFormerBasedCompressorV3(query_seqlen=query_seqlen, model_dim=hidden_size, keep_ratio=keep_ratio)
        elif compressor_version == 2:
            self.compressor = QFormerBasedCompressorV2(query_seqlen=query_seqlen, model_dim=hidden_size)
        else:
            self.compressor = QFormerBasedCompressor(query_seqlen=query_seqlen, model_dim=hidden_size)
        self.decompressor = QFormerDecompressor(hidden_size)
        self.fuser = nn.Linear(hidden_size * 2, hidden_size)
        self.initialize_weights()
        

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        if hasattr(self.compressor, 'query'):          # V1
            nn.init.normal_(self.compressor.query, 0, .02)
        if hasattr(self.compressor, '_zero_adaptive'):  # V2: re-zero FiLM/importance after xavier apply
            self.compressor._zero_adaptive()
        # decompressor.pos_query is trunc_normal_-initialised in its own __init__

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        b, _, h, w = x.shape
        hx = rearrange(x, "b c (h p) (w q) -> (b h w) c p q", p=self.patch_size, q=self.patch_size)
        hx = rearrange(self.pool(hx), "(b h w) c p q -> b (h p w q) c", h=h//self.patch_size, w=w//self.patch_size)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        aux_recon_loss = None
        for i, block in enumerate(self.blocks):
            if i == self.layer_landmarks[0]:
                r = x
                B, L, D = x.shape
                x = self.compressor(x)
                # [aux #1] decode r straight back from the compressed tokens (bypassing
                # the transformer blocks) so compressor+decompressor form an autoencoder
                # pair whose bottleneck must retain reconstructable detail. Target is
                # detached: preserve whatever the early blocks produced, don't collapse r
                # into something trivially compressible. Train-only (no sampling cost).
                if self.training and self.aux_recon:
                    aux_recon_loss = self.aux_recon_weight * nn.functional.mse_loss(
                        self.decompressor(x, L), r.detach())

            if i == self.layer_landmarks[1]:
                x = self.decompressor(x, L)
                x = torch.cat([r, x], dim=-1)
                x = self.fuser(x)
            x = block(x, c)                      # (N, T, D)
        x = self.detailer(torch.cat([x, hx], dim=-1))
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if aux_recon_loss is not None:
            return x, aux_recon_loss
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}
