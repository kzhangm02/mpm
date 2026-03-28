import torch
import numpy as np

def conv2d_relu(inch, outch, kernel_size, padding, stride=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(
            inch, outch, kernel_size=kernel_size, 
            padding=padding, stride=stride
        ),
        torch.nn.ReLU(),
    )
    return convlayer

def deconv_relu(inch, outch, kernel_size, stride, padding, output_padding=0):
    deconvlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            inch, outch, kernel_size=kernel_size, stride=stride, 
            padding=padding, output_padding=output_padding,
        ),
        torch.nn.ReLU(),
    )
    return deconvlayer

def deconv(inch, outch, kernel_size, stride, padding, output_padding=0):
    deconvlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            inch, outch, kernel_size=kernel_size, stride=stride, 
            padding=padding, output_padding=output_padding,
        )
    )
    return deconvlayer

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).cuda()

def unstack_and_split(x, batch_size, num_channels=3):
    unstack_shape = [batch_size, -1] + list(x.shape[1:])
    unstacked = torch.reshape(x, unstack_shape)
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=-1)
    masks = masks.softmax(dim=1)
    return channels, masks


class SoftPositionEmbed(torch.nn.Module):
    def __init__(self, hidden_size, resolution):
        super(SoftPositionEmbed, self).__init__()
        self.embedding = torch.nn.Linear(4, hidden_size)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class ClusterSlotAttention(torch.nn.Module):
    def __init__(
        self,
        num_slots,
        num_iterations,
        num_slot_clusters,
        slot_size=64, 
        input_size=64
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
        

    def forward(self, inputs):
        num_inputs = inputs.shape[1]
        inputs = self.norm_inputs(inputs.reshape(-1, self.input_size))
        inputs = inputs.reshape(-1, num_inputs, self.input_size)
        # `slots` has shape: [batch_size, num_slots, slot_size]
        slots = self.slots_init.repeat(inputs.shape[0], 1, 1)

        for i in range(self.num_iterations):
            # E-step p(z|x,mu)
            k = inputs.unsqueeze(2) - 0.5 * slots.unsqueeze(1)
            likelihood_attn_logits = torch.einsum('ijkl, ikl -> ijk', k, slots)
            likelihood_attn = likelihood_attn_logits.softmax(dim=-1) + self.epsilon
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
        
        attns = (likelihood_attn, prior_attn)
        return slots, attns

class SlotAttentionAutoencoder(torch.nn.Module):
    def __init__(
        self,
        resolution,
        num_slots,
        num_iterations,
        num_slot_clusters,
        additive_decoder,
    ):
        super(SlotAttentionAutoencoder, self).__init__()

        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_slot_clusters = num_slot_clusters
        self.additive_decoder = additive_decoder
        
        self.conv_stack1 = conv2d_relu(resolution[-1], 64, 5, padding=2)
        self.conv_stack2 = conv2d_relu(64, 64, 5, padding=2)
        self.conv_stack3 = conv2d_relu(64, 64, 5, padding=2)
        self.conv_stack4 = conv2d_relu(64, 64, 5, padding=2)
        self.conv_stack5 = conv2d_relu(64, 64, 3, padding=1)
        self.conv_stack6 = conv2d_relu(64, 64, 3, padding=1)
        self.encoder_pos = SoftPositionEmbed(64, resolution[:-1])

        self.deconv_6 = deconv_relu(64, 256, 5, stride=1, padding=0)       # -> 5x5
        self.deconv_5 = deconv_relu(256, 128, 5, stride=2, padding=2)      # -> 9x9
        self.deconv_4 = deconv_relu(128, 128, 5, stride=2, padding=2)      # -> 17x17
        self.deconv_3 = deconv_relu(128, 64, 5, stride=2, padding=2)       # -> 33x33
        self.deconv_2 = deconv_relu(64, 64, 5, stride=1, padding=1)        # -> 35x35
        self.deconv_1 = deconv(64, 4, 3, stride=1, padding=1)              # -> 35x35

        self.enc_mlp = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
        )
        self.dec_mlp = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
        )
        self.slot_attention = ClusterSlotAttention(
            num_slots=self.num_slots,
            num_iterations=self.num_iterations,
            num_slot_clusters=self.num_slot_clusters,
        )

    def encoder(self, x):
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0,3,1,2)
        x = self.conv_stack1(x)
        x = self.conv_stack2(x)
        x = self.conv_stack3(x)
        x = self.conv_stack4(x)
        x = self.conv_stack5(x)
        x = self.conv_stack6(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        return x

    def decoder(self, x):
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0,3,1,2)
        x = self.deconv_6(x)
        x = self.deconv_5(x)
        x = self.deconv_4(x)
        x = self.deconv_3(x)
        x = self.deconv_2(x)
        x = self.deconv_1(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0,2,3,1)
        return x

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        x = self.enc_mlp(x)
        x = torch.flatten(x, 1, 2)
        # x shape: [batch_size, height * width, input_size]
        slots, attns = self.slot_attention(x)
            
        x = slots + self.dec_mlp(slots)
        x = x.reshape(-1, 1, 1, 64)
        x = self.decoder(x)

        if self.additive_decoder:  
            recons, masks = unstack_and_split(x, batch_size)
            combined = torch.sum(recons * masks, axis=1)
        else:
            combined = None
            recons, masks = unstack_and_split(x, batch_size)
            recons = torch.clamp(recons * masks, min=0.0, max=1.0)

        return combined, recons, masks, slots, attns