import torch
import numpy as np
from transformers import BertModel   # type: ignore


def segment_avg(x, mask):
    B, N, D = x.shape
    device = x.device

    # segment IDs: 0,0,...,1,1,...,2,... etc.
    seg_ids = mask.cumsum(dim=1) - 1                       # (B, N)
    num_segs = mask.sum(dim=1).max().item()
    seg_ids_exp = seg_ids.unsqueeze(-1).expand(-1, -1, D)  # (B, N, D)
    seg_sums = torch.zeros(B, num_segs, D, device=device)
    seg_sums.scatter_add_(1, seg_ids_exp, x)
    counts = torch.zeros(B, num_segs, 1, device=device)
    counts.scatter_add_(1, seg_ids.unsqueeze(-1), torch.ones(B, N, 1, device=device))

    seg_means = seg_sums / counts
    out = torch.gather(seg_means, 1, seg_ids_exp)
    return out


class TextClusterSlotAttention(torch.nn.Module):
    def __init__(
        self,
        slot_size,
        input_size,
        num_slots,
        num_iterations,
        num_slot_clusters,
    ):
        super().__init__()
        self.slot_size = slot_size
        self.input_size = input_size
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_slot_clusters = num_slot_clusters
        self.epsilon = 1e-8

        self.norm_inputs = torch.nn.BatchNorm1d(self.input_size)
        self.slots_init = torch.nn.Parameter(torch.randn(1, self.num_slots, self.slot_size))
        if self.num_slot_clusters is not None and self.num_slot_clusters > 0:
            centers_shape = (1, self.num_slot_clusters, self.slot_size)
            self.centers = torch.nn.Parameter(torch.randn(centers_shape))
            self.global_clustering = True
        else:
            self.global_clustering = False
        print('global clustering:', self.global_clustering)
        
    def forward(self, inputs, masks, word_starts):
        num_inputs = inputs.shape[1]
        word_avg = True
        if word_avg:
            inputs = segment_avg(inputs, word_starts)
            inputs = self.norm_inputs(inputs.reshape(-1, self.input_size))
            inputs = inputs.reshape(-1, num_inputs, self.input_size)
            padding_masks = torch.clone(masks)
            masks = masks * word_starts
        else:
            inputs = self.norm_inputs(inputs.reshape(-1, self.input_size))
            inputs = inputs.reshape(-1, num_inputs, self.input_size)

        # `slots` has shape: [batch_size, num_topics, slot_size]
        slots = self.slots_init.repeat(inputs.shape[0], 1, 1)

        for i in range(self.num_iterations):
            # E-step p(z|x,mu)
            k = inputs.unsqueeze(2) - 0.5 * slots.unsqueeze(1)
            likelihood_attn_logits = torch.einsum('ijkl, ikl -> ijk', k, slots)
            likelihood_attn = likelihood_attn_logits.softmax(dim=-1) + self.epsilon
            # for different lengths
            if word_avg:
                pad_masked_likelihood_attn = likelihood_attn * padding_masks.unsqueeze(2) 
            likelihood_attn = likelihood_attn * masks.unsqueeze(2)
            likelihood_attn_norm = likelihood_attn.sum(dim=1).unsqueeze(2)

            if self.global_clustering:
                # GMM prior
                flat_slots = slots.reshape(-1, self.slot_size)
                prior_k = self.centers - 0.5 * flat_slots.unsqueeze(1)
                prior_attn_logits = torch.einsum('ijk, ik -> ij', prior_k, flat_slots)
                prior_attn = prior_attn_logits.softmax(dim=-1) + self.epsilon
                prior_attn_norm = prior_attn.sum(dim=1).reshape(-1, self.num_slots, 1)

                # M-step
                likelihood_slots = torch.einsum('ijk, ijl -> ikl', likelihood_attn, inputs)
                prior_slots = torch.einsum('ij, jk -> ik', prior_attn, self.centers[0])
                prior_slots = prior_slots.reshape(-1, self.num_slots, self.slot_size)
                attn_norm = likelihood_attn_norm + prior_attn_norm
                slots = (likelihood_slots + prior_slots) / attn_norm
            else:
                # M-step
                slots = torch.einsum('ijk, ijl -> ikl', likelihood_attn, inputs)
                slots /= likelihood_attn_norm
                prior_attn = None
        
        if word_avg:
            attns = (pad_masked_likelihood_attn, prior_attn)
        else:
            attns = (likelihood_attn, prior_attn)
        return slots, attns


class TextSlotAttentionAutoencoder(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        slot_size,
        num_slots,
        num_iterations,
        num_slot_clusters,  
    ):
        super(TextSlotAttentionAutoencoder, self).__init__()
        self.tokenizer = tokenizer
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_slot_clusters = num_slot_clusters
        
        if self.tokenizer == 'bert':
            self.hidden_dim = 768
            self.vocab_size = 28996
            self.encoder = BertModel.from_pretrained("bert-base-uncased")
            freeze_bert = True
            if freeze_bert:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            raise ValueError(f"{self.tokenizer} not implemented")
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.slot_size, self.slot_size), torch.nn.ReLU(),
            torch.nn.Linear(self.slot_size, self.slot_size), torch.nn.ReLU(),
            torch.nn.Linear(self.slot_size, self.vocab_size),
        )
        self.enc_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.slot_size),
        )
        self.dec_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.slot_size, self.slot_size), torch.nn.ReLU(),
            torch.nn.Linear(self.slot_size, self.slot_size),
        )
        self.slot_attention = TextClusterSlotAttention(
            slot_size=self.slot_size,
            input_size=self.slot_size,
            num_slots=self.num_slots,
            num_iterations=self.num_iterations,
            num_slot_clusters=self.num_slot_clusters,
        )
        self.N = 512
        self.pos_enc = torch.nn.Parameter(
            torch.randn(1, self.N, 1, self.slot_size)
        )

    def forward(self, x, masks, word_starts):
        batch_size = x.shape[0]
        x = self.encoder(x, attention_mask=masks)
        x = self.enc_mlp(x.last_hidden_state)
        # x shape: [batch_size, seq_len, hidden_dim]
        slots, attns = self.slot_attention(x, masks, word_starts)
        x = slots + self.dec_mlp(slots)
        logits = self.decoder(x.unsqueeze(1) + self.pos_enc)
        # logits shape: [batch_size, num_topics, vocab_size]
        return logits, slots, attns