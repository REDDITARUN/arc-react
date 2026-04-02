import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_COLORS = 10
dim = 128
IGNORE_INDEX = -100


def make_safe_padding_mask(valid_mask):
    padding_mask = ~valid_mask.bool()
    fully_masked = padding_mask.all(dim=1)

    if fully_masked.any():
        padding_mask = padding_mask.clone()
        padding_mask[fully_masked, 0] = False

    return padding_mask

#  From scripts/data_loader build_task_sample, one sample has:

#   - example_inputs: [Kmax, 30, 30]
#   - example_outputs: [Kmax, 30, 30]
#   - example_input_masks: [Kmax, 30, 30]
#   - example_output_masks: [Kmax, 30, 30]
#   - query_input: [30, 30]
#   - query_output: [30, 30]
#   - example_slot_mask: [Kmax]

#   After collation in arc_collate_fn, a batch is:

#   - example_inputs: [B, Kmax, 30, 30]
#   - example_outputs: [B, Kmax, 30, 30]
#   - example_input_masks: [B, Kmax, 30, 30]
#   - example_output_masks: [B, Kmax, 30, 30]
#   - query_input: [B, 30, 30]
#   - query_output: [B, 30, 30]
#   - query_input_mask: [B, 30, 30]
#   - example_slot_mask: [B, Kmax]

#   So for your GridEncoder, the direct input should be:

#   - grid: [B, 30, 30], dtype=torch.long
#   - mask: [B, 30, 30], usually float/bool


class GridEncoder(nn.Module):
    def __init__(
        self, d_model, 
        num_colors=NUM_COLORS,
        grid_size=30,
        num_layers=3,
        nhead=8,
        dropout=0.1,
    ):
        super().__init__()
        self.grid_size = grid_size
        # Learn a vector for each color ID in the grid.
        # If num_colors=10 and d_model=128, this embedding table has shape [10, 128].
        # Example:
        #   color 0 -> 128-dim learnable vector
        #   color 1 -> 128-dim learnable vector
        #   ...
        #   color 9 -> 128-dim learnable vector
        # When we pass a grid of color IDs into self.color_emb, each cell value
        # is replaced by its embedding vector.
        self.color_emb = nn.Embedding(num_colors, d_model)
        self.validity_emb = nn.Embedding(2, d_model)
        # similar for row and column embeddings
        self.row_emb = nn.Embedding(grid_size, d_model)
        self.col_emb = nn.Embedding(grid_size, d_model)

        # One encoder layer, then stack it num_layers times.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,   # expects [B, seq_len, d_model]
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, grid, mask, attention_mask=None):
        # grid: [B, H, W] int
        B = grid.shape[0]
        H = grid.shape[1]
        W = grid.shape[2]
        # create row and column indices
        r = torch.arange(H, device=grid.device)
        c = torch.arange(W, device=grid.device)

        # embed row and column indices
        row = self.row_emb(r)[None, :, None, :]
        col = self.col_emb(c)[None, None, :, :]

        validity = mask.bool()
        x = self.color_emb(grid) + self.validity_emb(validity.long()) + row + col  # [B,H,W,d]
        x = x.reshape(B, H * W, -1)

        observed_mask = validity.reshape(B, H * W)
        if attention_mask is None:
            attention_mask = mask
        attn_mask = attention_mask.reshape(B, H * W).bool()
        padding_mask = make_safe_padding_mask(attn_mask)

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = x.masked_fill((~observed_mask).unsqueeze(-1), 0.0)

        return x, observed_mask  # [B,900,d], [B,900]


class ObjectExtractor(nn.Module):
    def __init__(self, max_objects=16, shape_pool=5, num_colors=NUM_COLORS):
        super().__init__()
        self.max_objects = max_objects
        self.shape_pool = shape_pool
        self.num_colors = num_colors
        self.feature_dim = 11 + shape_pool * shape_pool

    @torch.no_grad()
    def _extract_single(self, grid, mask):
        # grid, mask: [H, W]
        device = grid.device
        grid_cpu = grid.detach().cpu()
        mask_cpu = mask.detach().cpu().bool()

        H, W = grid_cpu.shape
        valid_colors = grid_cpu[mask_cpu]
        if valid_colors.numel() == 0:
            bg = 0
        else:
            bg = torch.bincount(valid_colors, minlength=self.num_colors).argmax().item()

        visited = torch.zeros(H, W, dtype=torch.bool)
        objects = []

        for r in range(H):
            for c in range(W):
                if not mask_cpu[r, c]:
                    continue
                if visited[r, c]:
                    continue

                color = int(grid_cpu[r, c].item())
                if color == bg:
                    visited[r, c] = True
                    continue

                # BFS connected component, same color, 4-connected
                queue = [(r, c)]
                visited[r, c] = True
                coords = []

                while queue:
                    rr, cc = queue.pop()
                    coords.append((rr, cc))

                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if nr < 0 or nr >= H or nc < 0 or nc >= W:
                            continue
                        if visited[nr, nc]:
                            continue
                        if not mask_cpu[nr, nc]:
                            continue
                        if int(grid_cpu[nr, nc].item()) != color:
                            continue
                        visited[nr, nc] = True
                        queue.append((nr, nc))

                if not coords:
                    continue

                rows = [p[0] for p in coords]
                cols = [p[1] for p in coords]

                r0, r1 = min(rows), max(rows)
                c0, c1 = min(cols), max(cols)

                bh = r1 - r0 + 1
                bw = c1 - c0 + 1
                area = bh * bw
                size = len(coords)
                fill_ratio = size / max(area, 1)

                cy = sum(rows) / size
                cx = sum(cols) / size

                aspect = bw / max(bh, 1)

                crop = torch.zeros((bh, bw), dtype=torch.float32)
                for rr, cc in coords:
                    crop[rr - r0, cc - c0] = 1.0

                pooled = F.adaptive_avg_pool2d(
                    crop.unsqueeze(0).unsqueeze(0),
                    (self.shape_pool, self.shape_pool),
                ).view(-1)

                feat = torch.tensor(
                    [
                        r0 / max(H - 1, 1),
                        c0 / max(W - 1, 1),
                        r1 / max(H - 1, 1),
                        c1 / max(W - 1, 1),
                        bh / max(H, 1),
                        bw / max(W, 1),
                        cy / max(H - 1, 1),
                        cx / max(W - 1, 1),
                        size / max(H * W, 1),
                        fill_ratio,
                        aspect,
                    ],
                    dtype=torch.float32,
                )

                feat = torch.cat([feat, pooled], dim=0)
                objects.append((size, color, feat))

        objects.sort(key=lambda x: x[0], reverse=True)
        objects = objects[: self.max_objects]

        feats = torch.zeros(self.max_objects, self.feature_dim, dtype=torch.float32)
        colors = torch.zeros(self.max_objects, dtype=torch.long)
        obj_mask = torch.zeros(self.max_objects, dtype=torch.bool)

        for i, (_, color, feat) in enumerate(objects):
            feats[i] = feat
            colors[i] = color
            obj_mask[i] = True

        return feats.to(device), colors.to(device), obj_mask.to(device)

    @torch.no_grad()
    def forward(self, grids, masks):
        # grids, masks: [B, H, W]
        B = grids.size(0)
        feat_list, color_list, mask_list = [], [], []

        for b in range(B):
            feats, colors, obj_mask = self._extract_single(grids[b], masks[b])
            feat_list.append(feats)
            color_list.append(colors)
            mask_list.append(obj_mask)

        feats = torch.stack(feat_list, dim=0)     # [B, M, F]
        colors = torch.stack(color_list, dim=0)   # [B, M]
        obj_mask = torch.stack(mask_list, dim=0)  # [B, M]
        return feats, colors, obj_mask


class ObjectEncoder(nn.Module):
    def __init__(self, d_model, max_objects=16, shape_pool=5, num_colors=NUM_COLORS):
        super().__init__()
        self.max_objects = max_objects
        self.feature_dim = 11 + shape_pool * shape_pool

        self.feat_mlp = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.color_emb = nn.Embedding(num_colors, d_model)
        self.role_emb = nn.Embedding(3, d_model)   # 0=input, 1=output, 2=query
        self.slot_emb = nn.Embedding(max_objects, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats, colors, obj_mask, role_id):
        # feats: [B, M, F], colors: [B, M], obj_mask: [B, M]
        B, M, _ = feats.shape
        device = feats.device

        slot_ids = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
        role_ids = torch.full((B, M), role_id, device=device, dtype=torch.long)

        x = (
            self.feat_mlp(feats)
            + self.color_emb(colors)
            + self.role_emb(role_ids)
            + self.slot_emb(slot_ids)
        )
        x = self.norm(x)
        x = x.masked_fill((~obj_mask).unsqueeze(-1), 0.0)
        return x, obj_mask



class PairReasoner(nn.Module):
    def __init__(self, d_model, nhead=8, num_rule_tokens=4, dropout=0.1):
        super().__init__()
        self.ie_to_oe = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.oe_to_ie = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.fuse = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.rule_tokens = nn.Parameter(torch.randn(1, num_rule_tokens, d_model))
        self.rule_to_input_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.rule_to_output_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.rule_to_fused_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, ie_tokens, oe_tokens, ie_mask, oe_mask):
        # ie_tokens, oe_tokens: [B, 900, d]
        # ie_mask, oe_mask: [B, 900] where True = valid
        ie_pad = make_safe_padding_mask(ie_mask)
        oe_pad = make_safe_padding_mask(oe_mask)

        ie_ctx, _ = self.ie_to_oe(
            query=ie_tokens, key=oe_tokens, value=oe_tokens,
            key_padding_mask=oe_pad
        )
        oe_ctx, _ = self.oe_to_ie(
            query=oe_tokens, key=ie_tokens, value=ie_tokens,
            key_padding_mask=ie_pad
        )

        fused = torch.cat([
            ie_tokens,
            oe_tokens,
            ie_ctx,
            oe_ctx,
        ], dim=-1)
        fused = self.fuse(fused)  # [B, 900, d]

        B = fused.size(0)
        rule_tokens = self.rule_tokens.expand(B, -1, -1)  # [B, R, d]
        rule_from_input, _ = self.rule_to_input_attn(
            query=rule_tokens,
            key=ie_tokens,
            value=ie_tokens,
            key_padding_mask=make_safe_padding_mask(ie_mask),
        )
        rule_from_output, _ = self.rule_to_output_attn(
            query=rule_tokens,
            key=oe_tokens,
            value=oe_tokens,
            key_padding_mask=make_safe_padding_mask(oe_mask),
        )
        fused_mask = ie_mask | oe_mask
        rule_from_fused, _ = self.rule_to_fused_attn(
            query=rule_tokens,
            key=fused,
            value=fused,
            key_padding_mask=make_safe_padding_mask(fused_mask),
        )
        pair_rule_tokens = self.out_norm(rule_tokens + rule_from_input + rule_from_output + rule_from_fused)
        return pair_rule_tokens

class GlobalReasoner(nn.Module):
    def __init__(self, d_model, nhead=8, num_global_tokens=4, dropout=0.1):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, d_model))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, ie_tokens, oe_tokens, ie_mask, oe_mask, example_mask):
        # ie_tokens, oe_tokens: [B, K, N, d]
        # ie_mask, oe_mask: [B, K, N]  True = valid token
        # example_mask: [B, K]         True = valid example
        B, K, N, D = ie_tokens.shape

        pair_tokens = torch.cat([ie_tokens, oe_tokens], dim=-1)   # [B,K,N,2d]
        pair_tokens = self.fuse(pair_tokens)                      # [B,K,N,d]

        example_token_mask = example_mask.unsqueeze(-1)
        input_memory_mask = ie_mask & example_token_mask
        output_memory_mask = oe_mask & example_token_mask

        memory = torch.cat([ie_tokens, oe_tokens, pair_tokens], dim=2)              # [B,K,3N,d]
        memory_mask = torch.cat(
            [input_memory_mask, output_memory_mask, input_memory_mask | output_memory_mask],
            dim=2,
        )                                                                           # [B,K,3N]

        memory = memory.reshape(B, 3 * K * N, D)                                    # [B,3KN,d]
        memory_mask = memory_mask.reshape(B, 3 * K * N)                             # [B,3KN]
        padding_mask = make_safe_padding_mask(memory_mask)        # True = ignore

        global_tokens = self.global_tokens.expand(B, -1, -1)      # [B,G,d]

        attn_out, _ = self.cross_attn(
            query=global_tokens,
            key=memory,
            value=memory,
            key_padding_mask=padding_mask,
        )

        global_tokens = self.norm(global_tokens + attn_out)
        global_tokens = global_tokens + self.ff(global_tokens)

        return global_tokens, memory, memory_mask


class RuleCreator(nn.Module):
    def __init__(self, d_model, nhead=8, num_rule_tokens=4, num_hypotheses=4, dropout=0.1):
        super().__init__()
        self.num_hypotheses = num_hypotheses
        self.num_rule_tokens = num_rule_tokens

        self.rule_tokens = nn.Parameter(
            torch.randn(1, num_hypotheses, num_rule_tokens, d_model)
        )

        self.pair_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.global_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.query_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        pair_rule_tokens,
        example_mask,
        global_tokens,
        query_tokens,
        query_mask,
    ):
        # pair_rule_tokens: [B, K, R, d]
        B, K, R, D = pair_rule_tokens.shape
        HYP = self.num_hypotheses
        RR = self.num_rule_tokens

        pair_memory = pair_rule_tokens.reshape(B, K * R, D)
        pair_memory_mask = example_mask.unsqueeze(-1).expand(B, K, R).reshape(B, K * R)

        rule_tokens = self.rule_tokens.expand(B, -1, -1, -1)   # [B, HYP, RR, d]
        rule_tokens = rule_tokens.reshape(B * HYP, RR, D)

        pair_memory_rep = pair_memory.unsqueeze(1).expand(B, HYP, K * R, D).reshape(B * HYP, K * R, D)
        pair_mask_rep = pair_memory_mask.unsqueeze(1).expand(B, HYP, K * R).reshape(B * HYP, K * R)

        global_rep = global_tokens.unsqueeze(1).expand(B, HYP, global_tokens.size(1), D).reshape(B * HYP, global_tokens.size(1), D)

        query_rep = query_tokens.unsqueeze(1).expand(B, HYP, query_tokens.size(1), D).reshape(B * HYP, query_tokens.size(1), D)
        query_mask_rep = query_mask.unsqueeze(1).expand(B, HYP, query_mask.size(1)).reshape(B * HYP, query_mask.size(1))

        pair_ctx, _ = self.pair_attn(
            query=rule_tokens,
            key=pair_memory_rep,
            value=pair_memory_rep,
            key_padding_mask=make_safe_padding_mask(pair_mask_rep),
        )
        rule_tokens = self.norm1(rule_tokens + pair_ctx)

        global_ctx, _ = self.global_attn(
            query=rule_tokens,
            key=global_rep,
            value=global_rep,
        )
        rule_tokens = self.norm2(rule_tokens + global_ctx)

        query_ctx, _ = self.query_attn(
            query=rule_tokens,
            key=query_rep,
            value=query_rep,
            key_padding_mask=make_safe_padding_mask(query_mask_rep),
        )
        rule_tokens = self.norm3(rule_tokens + query_ctx)

        rule_tokens = self.norm4(rule_tokens + self.ff(rule_tokens))
        rule_tokens = rule_tokens.reshape(B, HYP, RR, D)
        return rule_tokens


class QuerySolver(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()

        self.rule_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.refine = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, query_tokens, query_mask, rule_tokens):
        # query_tokens: [B, N, d]
        # query_mask:   [B, N] where True = valid
        # rule_tokens:  [B, R, d]

        query_ctx, _ = self.rule_attn(
            query=query_tokens,
            key=rule_tokens,
            value=rule_tokens,
        )

        x = self.norm(query_tokens + query_ctx)

        padding_mask = make_safe_padding_mask(query_mask)
        x = self.refine(x, src_key_padding_mask=padding_mask)
        x = x.masked_fill((~query_mask).unsqueeze(-1), 0.0)

        return x  # predicted_output_tokens: [B, N, d]


# OutputHead class
class OutputHead(nn.Module):
    def __init__(self, d_model, num_colors=NUM_COLORS, grid_size=30):
        super().__init__()
        self.num_colors = num_colors
        self.grid_size = grid_size
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, num_colors)

    def forward(self, output_tokens, output_mask=None):
        # output_tokens: [B, N, d]
        # output_mask: [B, N] where True = valid token
        B, N, D = output_tokens.shape

        x = self.norm(output_tokens)
        logits = self.proj(x)  # [B, N, num_colors]

        if output_mask is not None:
            logits = logits.masked_fill((~output_mask).unsqueeze(-1), 0.0)

        logits = logits.reshape(B, self.grid_size, self.grid_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2).contiguous()  # [B, num_colors, H, W]
        return logits

class RulePCN(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=8,
        num_steps=3,
        step_size=0.5,
        energy_weight_pair=1.0,
        energy_weight_global=1.0,
        energy_weight_query=1.0,
        dropout=0.1,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.step_size = step_size
        self.energy_weight_pair = energy_weight_pair
        self.energy_weight_global = energy_weight_global
        self.energy_weight_query = energy_weight_query

        self.pair_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.global_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.query_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.delta_net = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.pair_energy_proj = nn.Linear(d_model, d_model)
        self.global_energy_proj = nn.Linear(d_model, d_model)
        self.query_energy_proj = nn.Linear(d_model, d_model)

        self.out_norm = nn.LayerNorm(d_model)

    def _attend_once(
        self,
        x,                  # [B,HYP,R,d]
        pair_rule_tokens,   # [B,K,Rp,d]
        example_mask,       # [B,K]
        global_tokens,      # [B,G,d]
        query_tokens,       # [B,N,d]
        query_mask,         # [B,N]
    ):
        B, HYP, R, D = x.shape
        _, K, Rp, _ = pair_rule_tokens.shape
        G = global_tokens.size(1)
        N = query_tokens.size(1)

        x_flat = x.reshape(B * HYP, R, D)

        pair_memory = pair_rule_tokens.reshape(B, K * Rp, D)
        pair_mask = example_mask.unsqueeze(-1).expand(B, K, Rp).reshape(B, K * Rp)
        pair_memory = pair_memory.unsqueeze(1).expand(B, HYP, K * Rp, D).reshape(B * HYP, K * Rp, D)
        pair_mask = pair_mask.unsqueeze(1).expand(B, HYP, K * Rp).reshape(B * HYP, K * Rp)

        global_rep = global_tokens.unsqueeze(1).expand(B, HYP, G, D).reshape(B * HYP, G, D)
        query_rep = query_tokens.unsqueeze(1).expand(B, HYP, N, D).reshape(B * HYP, N, D)
        query_mask_rep = query_mask.unsqueeze(1).expand(B, HYP, N).reshape(B * HYP, N)

        pair_ctx, _ = self.pair_attn(
            query=x_flat,
            key=pair_memory,
            value=pair_memory,
            key_padding_mask=make_safe_padding_mask(pair_mask),
        )
        global_ctx, _ = self.global_attn(
            query=x_flat,
            key=global_rep,
            value=global_rep,
        )
        query_ctx, _ = self.query_attn(
            query=x_flat,
            key=query_rep,
            value=query_rep,
            key_padding_mask=make_safe_padding_mask(query_mask_rep),
        )

        pair_ctx = pair_ctx.reshape(B, HYP, R, D)
        global_ctx = global_ctx.reshape(B, HYP, R, D)
        query_ctx = query_ctx.reshape(B, HYP, R, D)
        return pair_ctx, global_ctx, query_ctx

    def compute_energy(
        self,
        rule_tokens,
        pair_rule_tokens,
        example_mask,
        global_tokens,
        query_tokens,
        query_mask,
    ):
        pair_ctx, global_ctx, query_ctx = self._attend_once(
            rule_tokens, pair_rule_tokens, example_mask, global_tokens, query_tokens, query_mask
        )

        e_pair = ((rule_tokens - self.pair_energy_proj(pair_ctx)) ** 2).mean(dim=(-1, -2))
        e_global = ((rule_tokens - self.global_energy_proj(global_ctx)) ** 2).mean(dim=(-1, -2))
        e_query = ((rule_tokens - self.query_energy_proj(query_ctx)) ** 2).mean(dim=(-1, -2))

        total_energy = (
            self.energy_weight_pair * e_pair
            + self.energy_weight_global * e_global
            + self.energy_weight_query * e_query
        )  # [B, HYP]

        return {
            "energy_total_per_hyp": total_energy,
            "energy_pair_per_hyp": e_pair,
            "energy_global_per_hyp": e_global,
            "energy_query_per_hyp": e_query,
        }

    def infer(
        self,
        rule_tokens,
        pair_rule_tokens,
        example_mask,
        global_tokens,
        query_tokens,
        query_mask,
        return_trace=False,
    ):
        x = rule_tokens
        trace = []

        for _ in range(self.num_steps):
            pair_ctx, global_ctx, query_ctx = self._attend_once(
                x, pair_rule_tokens, example_mask, global_tokens, query_tokens, query_mask
            )
            delta_inp = torch.cat([x, pair_ctx, global_ctx, query_ctx], dim=-1)
            delta = self.delta_net(delta_inp)
            x = self.out_norm(x + self.step_size * delta)
            if return_trace:
                trace.append(x)

        energy_dict = self.compute_energy(
            x, pair_rule_tokens, example_mask, global_tokens, query_tokens, query_mask
        )

        out = {
            "rule_tokens_refined": x,
            **energy_dict,
        }
        if return_trace:
            out["trace"] = trace
        return out

class HypothesisScorer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, rule_tokens, pair_rule_tokens, global_tokens, query_tokens, query_mask):
        # rule_tokens: [B,HYP,R,d]
        B, HYP, R, D = rule_tokens.shape
        _, K, Rp, _ = pair_rule_tokens.shape

        rule_summary = rule_tokens.mean(dim=2)          # [B,HYP,d]

        pair_summary = pair_rule_tokens.mean(dim=(1, 2)).unsqueeze(1).expand(B, HYP, D)
        global_summary = global_tokens.mean(dim=1).unsqueeze(1).expand(B, HYP, D)

        qmask = query_mask.float()
        qden = qmask.sum(dim=1, keepdim=True).clamp_min(1.0)
        query_summary = (query_tokens * qmask.unsqueeze(-1)).sum(dim=1) / qden
        query_summary = query_summary.unsqueeze(1).expand(B, HYP, D)

        feat = torch.cat([rule_summary, pair_summary, global_summary, query_summary], dim=-1)
        scores = self.mlp(feat).squeeze(-1)   # [B,HYP]
        return scores



# ARCModel class
class ARCModel(nn.Module):
    def __init__(
        self,
        d_model=dim,
        num_colors=NUM_COLORS,
        grid_size=30,
        nhead=8,
        num_encoder_layers=3,
        num_query_layers=2,
        num_pair_rule_tokens=4,
        num_global_tokens=4,
        num_rule_tokens=4,
        dropout=0.1,
        # Hypotheses
        num_hypotheses=4,

        # Object encoder
        max_objects=16,
        object_shape_pool=5,

        # PCN
        pcn_num_steps=3,
        pcn_step_size=0.5,
        pcn_energy_weight=0.1,
    ):
        super().__init__()
        self.grid_encoder = GridEncoder(
            d_model=d_model,
            num_colors=num_colors,
            grid_size=grid_size,
            num_layers=num_encoder_layers,
            nhead=nhead,
            dropout=dropout,
        )
        self.pair_reasoner = PairReasoner(
            d_model=d_model,
            nhead=nhead,
            num_rule_tokens=num_pair_rule_tokens,
            dropout=dropout,
        )
        self.global_reasoner = GlobalReasoner(
            d_model=d_model,
            nhead=nhead,
            num_global_tokens=num_global_tokens,
            dropout=dropout,
        )
        self.rule_creator = RuleCreator(
            d_model=d_model,
            nhead=nhead,
            num_rule_tokens=num_rule_tokens,
            num_hypotheses=num_hypotheses,
            dropout=dropout,
        )
        self.hypothesis_scorer = HypothesisScorer(d_model=d_model)
        self.query_solver = QuerySolver(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_query_layers,
            dropout=dropout,
        )
        self.output_head = OutputHead(
            d_model=d_model,
            num_colors=num_colors,
            grid_size=grid_size,
        )
        self.num_colors = num_colors
        self.grid_size = grid_size

        self.pcn_energy_weight = pcn_energy_weight
        self.rule_pcn = RulePCN(
            d_model=d_model,
            nhead=nhead,
            num_steps=pcn_num_steps,
            step_size=pcn_step_size,
            dropout=dropout,
        )

        self.object_extractor = ObjectExtractor(
            max_objects=max_objects,
            shape_pool=object_shape_pool,
            num_colors=num_colors,
        )
        self.object_encoder = ObjectEncoder(
            d_model=d_model,
            max_objects=max_objects,
            shape_pool=object_shape_pool,
            num_colors=num_colors,
        )
        self.num_hypotheses = num_hypotheses

    def forward(
        self,
        example_inputs,
        example_outputs,
        example_input_masks,
        example_output_masks,
        query_input,
        query_input_mask,
        example_slot_mask,
        query_output_mask=None,
    ):
        # example_inputs: [B, K, H, W]
        # example_outputs: [B, K, H, W]
        # example_input_masks: [B, K, H, W]
        # example_output_masks: [B, K, H, W]
        # query_input: [B, H, W]
        # query_input_mask: [B, H, W]
        # query_output_mask: [B, H, W] or None
        # example_slot_mask: [B, K] where True = valid example slot

        B, K, H, W = example_inputs.shape

        flat_example_inputs = example_inputs.reshape(B * K, H, W)
        flat_example_outputs = example_outputs.reshape(B * K, H, W)
        flat_example_input_masks = example_input_masks.reshape(B * K, H, W)
        flat_example_output_masks = example_output_masks.reshape(B * K, H, W)

        # grid tokens for examples
        ie_grid_tokens, ie_grid_mask = self.grid_encoder(
            flat_example_inputs, flat_example_input_masks
        )
        oe_grid_tokens, oe_grid_mask = self.grid_encoder(
            flat_example_outputs, flat_example_output_masks
        )

        # object tokens for examples
        ie_obj_feats, ie_obj_colors, ie_obj_mask = self.object_extractor(
            flat_example_inputs, flat_example_input_masks
        )
        oe_obj_feats, oe_obj_colors, oe_obj_mask = self.object_extractor(
            flat_example_outputs, flat_example_output_masks
        )

        ie_obj_tokens, ie_obj_mask = self.object_encoder(
            ie_obj_feats, ie_obj_colors, ie_obj_mask, role_id=0
        )
        oe_obj_tokens, oe_obj_mask = self.object_encoder(
            oe_obj_feats, oe_obj_colors, oe_obj_mask, role_id=1
        )

        # hybrid example tokens = grid + objects
        ie_tokens = torch.cat([ie_grid_tokens, ie_obj_tokens], dim=1)
        oe_tokens = torch.cat([oe_grid_tokens, oe_obj_tokens], dim=1)
        ie_mask = torch.cat([ie_grid_mask, ie_obj_mask], dim=1)
        oe_mask = torch.cat([oe_grid_mask, oe_obj_mask], dim=1)

        N = ie_tokens.size(1)
        D = ie_tokens.size(2)

        ie_tokens = ie_tokens.reshape(B, K, N, D)
        oe_tokens = oe_tokens.reshape(B, K, N, D)
        ie_mask = ie_mask.reshape(B, K, N)
        oe_mask = oe_mask.reshape(B, K, N)

        pair_rule_tokens_list = []
        for k in range(K):
            pair_rule_k = self.pair_reasoner(
                ie_tokens[:, k],
                oe_tokens[:, k],
                ie_mask[:, k],
                oe_mask[:, k],
            )
            pair_rule_tokens_list.append(pair_rule_k)
        pair_rule_tokens = torch.stack(pair_rule_tokens_list, dim=1)  # [B, K, R, d]

        global_tokens, _, _ = self.global_reasoner(
            ie_tokens,
            oe_tokens,
            ie_mask,
            oe_mask,
            example_slot_mask.bool(),
        )

        # query tokens for rule formation: hybrid = grid + objects
        rule_query_grid_tokens, rule_query_grid_mask = self.grid_encoder(
            query_input,
            query_input_mask,
            attention_mask=query_input_mask,
        )

        query_obj_feats, query_obj_colors, query_obj_mask = self.object_extractor(
            query_input, query_input_mask
        )
        rule_query_obj_tokens, rule_query_obj_mask = self.object_encoder(
            query_obj_feats, query_obj_colors, query_obj_mask, role_id=2
        )

        rule_query_tokens = torch.cat([rule_query_grid_tokens, rule_query_obj_tokens], dim=1)
        query_observed_mask = torch.cat([rule_query_grid_mask, rule_query_obj_mask], dim=1)

        query_attention_mask = query_output_mask if query_output_mask is not None else query_input_mask
        solver_query_tokens, _ = self.grid_encoder(
            query_input,
            query_input_mask,
            attention_mask=query_attention_mask,
        )
        query_solver_mask = query_attention_mask.reshape(B, H * W).bool()

        rule_tokens = self.rule_creator(
            pair_rule_tokens=pair_rule_tokens,
            example_mask=example_slot_mask.bool(),
            global_tokens=global_tokens,
            query_tokens=rule_query_tokens,
            query_mask=query_observed_mask,
        )

        pcn_out = self.rule_pcn.infer(
            rule_tokens=rule_tokens,
            pair_rule_tokens=pair_rule_tokens,
            example_mask=example_slot_mask.bool(),
            global_tokens=global_tokens,
            query_tokens=rule_query_tokens,
            query_mask=query_observed_mask,
        )

        refined_rule_tokens = pcn_out["rule_tokens_refined"]

        hypothesis_scores = self.hypothesis_scorer(
            rule_tokens=refined_rule_tokens,
            pair_rule_tokens=pair_rule_tokens,
            global_tokens=global_tokens,
            query_tokens=rule_query_tokens,
            query_mask=query_observed_mask,
        )  # [B, HYP]



        B, HYP, RR, D = refined_rule_tokens.shape
        Nq = solver_query_tokens.size(1)

        solver_query_tokens_rep = solver_query_tokens.unsqueeze(1).expand(B, HYP, Nq, D).reshape(B * HYP, Nq, D)
        query_solver_mask_rep = query_solver_mask.unsqueeze(1).expand(B, HYP, Nq).reshape(B * HYP, Nq)
        refined_rule_tokens_rep = refined_rule_tokens.reshape(B * HYP, RR, D)

        output_tokens = self.query_solver(
            query_tokens=solver_query_tokens_rep,
            query_mask=query_solver_mask_rep,
            rule_tokens=refined_rule_tokens_rep,
        )

        logits = self.output_head(output_tokens, output_mask=query_solver_mask_rep)
        logits = logits.reshape(B, HYP, self.num_colors, H, W)

        return {
            "logits": logits,
            "pair_rule_tokens": pair_rule_tokens,
            "global_tokens": global_tokens,
            "rule_tokens_init": rule_tokens,
            "rule_tokens": refined_rule_tokens,
            "pcn_energy_per_hyp": pcn_out["energy_total_per_hyp"],
            "pcn_energy_pair_per_hyp": pcn_out["energy_pair_per_hyp"],
            "pcn_energy_global_per_hyp": pcn_out["energy_global_per_hyp"],
            "pcn_energy_query_per_hyp": pcn_out["energy_query_per_hyp"],
            "output_tokens": output_tokens,
            "query_mask": query_observed_mask,
            "output_mask": query_solver_mask_rep,
            "hypothesis_scores": hypothesis_scores,
        }


    def compute_loss(
        self,
        logits,              # [B,HYP,C,H,W]
        query_output,        # [B,H,W]
        query_output_mask,   # [B,H,W]
        hypothesis_scores=None,   # [B,HYP]
        pcn_energy=None,          # [B,HYP] or None
        pcn_energy_weight=0.0,
        ignore_index=IGNORE_INDEX,
    ):
        B, HYP, C, H, W = logits.shape

        target = query_output.clone().long()
        valid_mask = query_output_mask.bool()
        target = target.masked_fill(~valid_mask, ignore_index)

        target_rep = target.unsqueeze(1).expand(B, HYP, H, W).reshape(B * HYP, H, W)
        logits_rep = logits.reshape(B * HYP, C, H, W)

        ce_per = F.cross_entropy(
            logits_rep,
            target_rep,
            ignore_index=ignore_index,
            reduction="none",
        )  # [B*HYP,H,W]

        ce_per = ce_per.reshape(B, HYP, H, W)
        valid_mask_rep = valid_mask.unsqueeze(1).expand(B, HYP, H, W).float()
        ce_per_hyp = (ce_per * valid_mask_rep).sum(dim=(2, 3)) / valid_mask_rep.sum(dim=(2, 3)).clamp_min(1.0)

        total_per_hyp = ce_per_hyp
        if pcn_energy is not None:
            total_per_hyp = total_per_hyp + pcn_energy_weight * pcn_energy

        best_idx = total_per_hyp.argmin(dim=1)   # [B]
        loss = total_per_hyp.gather(1, best_idx.unsqueeze(1)).mean()

        with torch.no_grad():
            pred_per_hyp = logits.argmax(dim=2)  # [B,HYP,H,W]
            best_pred = pred_per_hyp[torch.arange(B, device=logits.device), best_idx]

            cell_correct = ((best_pred == query_output.long()) & valid_mask).sum().float()
            valid_cells = valid_mask.sum().clamp_min(1).float()
            cell_acc = cell_correct / valid_cells

            per_sample_correct = ((best_pred == query_output.long()) | (~valid_mask)).reshape(B, -1)
            grid_acc = per_sample_correct.all(dim=1).float().mean()

        metrics = {
            "loss": loss,
            "ce_loss": ce_per_hyp.gather(1, best_idx.unsqueeze(1)).mean().detach(),
            "cell_acc": cell_acc,
            "grid_acc": grid_acc,
            "best_hyp_idx": best_idx.detach(),
        }

        if pcn_energy is not None:
            metrics["pcn_energy"] = pcn_energy.gather(1, best_idx.unsqueeze(1)).mean().detach()

        if hypothesis_scores is not None:
            metrics["hypothesis_score_mean"] = hypothesis_scores.mean().detach()

        return metrics

    def training_step(self, batch):
        outputs = self(
            example_inputs=batch["example_inputs"],
            example_outputs=batch["example_outputs"],
            example_input_masks=batch["example_input_masks"],
            example_output_masks=batch["example_output_masks"],
            query_input=batch["query_input"],
            query_input_mask=batch["query_input_mask"],
            example_slot_mask=batch["example_slot_mask"],
            query_output_mask=batch["query_output_mask"],
        )

        metrics = self.compute_loss(
            logits=outputs["logits"],
            query_output=batch["query_output"],
            query_output_mask=batch["query_output_mask"],
            hypothesis_scores=outputs["hypothesis_scores"],
            pcn_energy=outputs["pcn_energy_per_hyp"],
            pcn_energy_weight=self.pcn_energy_weight,
        )

        return {
            **outputs,
            **metrics,
        }
