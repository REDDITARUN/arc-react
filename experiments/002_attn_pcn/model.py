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

#  From data_loader.py:138, one sample has:

#   - example_inputs: [Kmax, 30, 30]
#   - example_outputs: [Kmax, 30, 30]
#   - example_input_masks: [Kmax, 30, 30]
#   - example_output_masks: [Kmax, 30, 30]
#   - query_input: [30, 30]
#   - query_output: [30, 30]
#   - example_slot_mask: [Kmax]

#   After collation in data_loader.py:228, a batch is:

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


# New RuleCreator class
class RuleCreator(nn.Module):
    def __init__(self, d_model, nhead=8, num_rule_tokens=4, dropout=0.1):
        super().__init__()
        self.rule_tokens = nn.Parameter(torch.randn(1, num_rule_tokens, d_model))

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
        # example_mask: [B, K] where True = valid example
        # global_tokens: [B, G, d]
        # query_tokens: [B, N, d]
        # query_mask: [B, N] where True = valid query token
        B, K, R, D = pair_rule_tokens.shape

        pair_memory = pair_rule_tokens.reshape(B, K * R, D)               # [B, K*R, d]
        pair_memory_mask = example_mask.unsqueeze(-1).expand(B, K, R)
        pair_memory_mask = pair_memory_mask.reshape(B, K * R)             # [B, K*R]
        pair_padding_mask = make_safe_padding_mask(pair_memory_mask)

        rule_tokens = self.rule_tokens.expand(B, -1, -1)                  # [B, Rc, d]

        pair_ctx, _ = self.pair_attn(
            query=rule_tokens,
            key=pair_memory,
            value=pair_memory,
            key_padding_mask=pair_padding_mask,
        )
        rule_tokens = self.norm1(rule_tokens + pair_ctx)

        global_ctx, _ = self.global_attn(
            query=rule_tokens,
            key=global_tokens,
            value=global_tokens,
        )
        rule_tokens = self.norm2(rule_tokens + global_ctx)

        query_ctx, _ = self.query_attn(
            query=rule_tokens,
            key=query_tokens,
            value=query_tokens,
            key_padding_mask=make_safe_padding_mask(query_mask),
        )
        rule_tokens = self.norm3(rule_tokens + query_ctx)

        rule_tokens = self.norm4(rule_tokens + self.ff(rule_tokens))
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
        x,
        pair_rule_tokens,
        example_mask,
        global_tokens,
        query_tokens,
        query_mask,
    ):
        B, K, Rp, D = pair_rule_tokens.shape

        pair_memory = pair_rule_tokens.reshape(B, K * Rp, D)
        pair_memory_mask = example_mask.unsqueeze(-1).expand(B, K, Rp).reshape(B, K * Rp)

        pair_ctx, _ = self.pair_attn(
            query=x,
            key=pair_memory,
            value=pair_memory,
            key_padding_mask=make_safe_padding_mask(pair_memory_mask),
        )
        global_ctx, _ = self.global_attn(
            query=x,
            key=global_tokens,
            value=global_tokens,
        )
        query_ctx, _ = self.query_attn(
            query=x,
            key=query_tokens,
            value=query_tokens,
            key_padding_mask=make_safe_padding_mask(query_mask),
        )
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
            rule_tokens,
            pair_rule_tokens,
            example_mask,
            global_tokens,
            query_tokens,
            query_mask,
        )

        e_pair = ((rule_tokens - self.pair_energy_proj(pair_ctx)) ** 2).mean()
        e_global = ((rule_tokens - self.global_energy_proj(global_ctx)) ** 2).mean()
        e_query = ((rule_tokens - self.query_energy_proj(query_ctx)) ** 2).mean()

        total_energy = (
            self.energy_weight_pair * e_pair
            + self.energy_weight_global * e_global
            + self.energy_weight_query * e_query
        )

        return {
            "energy_total": total_energy,
            "energy_pair": e_pair,
            "energy_global": e_global,
            "energy_query": e_query,
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
                x,
                pair_rule_tokens,
                example_mask,
                global_tokens,
                query_tokens,
                query_mask,
            )

            delta_inp = torch.cat([x, pair_ctx, global_ctx, query_ctx], dim=-1)
            delta = self.delta_net(delta_inp)
            x = self.out_norm(x + self.step_size * delta)

            if return_trace:
                trace.append(x)

        energy_dict = self.compute_energy(
            x,
            pair_rule_tokens,
            example_mask,
            global_tokens,
            query_tokens,
            query_mask,
        )

        out = {
            "rule_tokens_refined": x,
            **energy_dict,
        }
        if return_trace:
            out["trace"] = trace
        return out

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
            dropout=dropout,
        )
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

        ie_tokens, ie_mask = self.grid_encoder(flat_example_inputs, flat_example_input_masks)
        oe_tokens, oe_mask = self.grid_encoder(flat_example_outputs, flat_example_output_masks)

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

        rule_query_tokens, query_observed_mask = self.grid_encoder(
            query_input,
            query_input_mask,
            attention_mask=query_input_mask,
        )
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

        output_tokens = self.query_solver(
            query_tokens=solver_query_tokens,
            query_mask=query_solver_mask,
            rule_tokens=refined_rule_tokens,
        )

        final_output_mask = query_solver_mask
        logits = self.output_head(output_tokens, output_mask=final_output_mask)

        return {
            "logits": logits,
            "pair_rule_tokens": pair_rule_tokens,
            "global_tokens": global_tokens,
            "rule_tokens_init": rule_tokens,
            "rule_tokens": refined_rule_tokens,
            "pcn_energy": pcn_out["energy_total"],
            "pcn_energy_pair": pcn_out["energy_pair"],
            "pcn_energy_global": pcn_out["energy_global"],
            "pcn_energy_query": pcn_out["energy_query"],
            "output_tokens": output_tokens,
            "query_mask": query_observed_mask,
            "output_mask": final_output_mask,
        }


    def compute_loss(
        self,
        logits,
        query_output,
        query_output_mask,
        pcn_energy=None,
        pcn_energy_weight=0.0,
        ignore_index=IGNORE_INDEX,
    ):
        # logits: [B, C, H, W]
        # query_output: [B, H, W]
        # query_output_mask: [B, H, W] where True/1 = valid target cell
        target = query_output.clone().long()
        valid_mask = query_output_mask.bool()
        target = target.masked_fill(~valid_mask, ignore_index)

        ce_loss = F.cross_entropy(logits, target, ignore_index=ignore_index)


        total_loss = ce_loss
        if pcn_energy is not None:
            total_loss = total_loss + pcn_energy_weight * pcn_energy

        with torch.no_grad():
            pred = logits.argmax(dim=1)  # [B, H, W]
            cell_correct = ((pred == query_output.long()) & valid_mask).sum().float()
            valid_cells = valid_mask.sum().clamp_min(1).float()
            cell_acc = cell_correct / valid_cells

            per_sample_correct = ((pred == query_output.long()) | (~valid_mask)).reshape(query_output.size(0), -1)
            grid_acc = per_sample_correct.all(dim=1).float().mean()

        metrics = {
            "loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "cell_acc": cell_acc,
            "grid_acc": grid_acc,
        }
        if pcn_energy is not None:
            metrics["pcn_energy"] = pcn_energy.detach()
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
            pcn_energy=outputs["pcn_energy"],
            pcn_energy_weight=self.pcn_energy_weight,
        )

        return {
            **outputs,
            **metrics,
        }
