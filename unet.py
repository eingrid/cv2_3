import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.projection = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, timesteps):
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :].to(timesteps.device)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.projection(embeddings)


class DoubleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        return x

class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_condition = False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.linear_time = nn.Linear(100, out_channels)
        self.use_condition = use_condition
        if use_condition:
            self.linear_condition = nn.Linear(64, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
            
    def forward(self, x, time_latent, condition = None):

        x = self.pool(x)
        x = self.conv1(x)
        x = self.norm1(x)
        if condition is not None and self.use_condition:
            condition_vector = self.linear_condition(condition)
            x = x + self.linear_time(time_latent)[..., None, None] + condition_vector[..., None, None]
        elif condition is None and self.use_condition == False:
            x = x + self.linear_time(time_latent)[..., None, None]
        else:
            raise ValueError("You can not use condition with this model, please check self.use_condition or do not pass condition")
        x = x + self.linear_time(time_latent)[..., None, None]
        x = F.gelu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        
        return x

class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_condition = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.linear_time = nn.Linear(100, out_channels)
        self.use_condition = use_condition
        if use_condition:
            self.linear_condition = nn.Linear(64, out_channels)
        
    def forward(self, x1, x2, time_latent, condition = None):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)

        if condition is not None and self.use_condition:
            condition_vector = self.linear_condition(condition)
            x = x + self.linear_time(time_latent)[..., None, None] + condition_vector[..., None, None]
        elif condition is None and self.use_condition == False:
            x = x + self.linear_time(time_latent)[..., None, None]
        else:
            raise ValueError("You can not use condition with this model, please check self.use_condition or do not pass condition")

        x = F.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        return x

class SelfAttentionLayer(nn.Module):
    """
    x - B,C,H,W
    condition - B,C
    """
    def __init__(self, channels, use_condition = False):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)
        self.use_condition = use_condition
        if use_condition:
            self.linear_condition = nn.Sequential(
                nn.Linear(64, channels),
            )
    def forward(self, x_input, condition = None):
        b, c, h, w = x_input.shape
        x = x_input.reshape(b, c, h * w).permute(0, 2, 1)
        #self attention here
        # basically we comput weighted average of x using tokens h*w, with c embedding size
        if condition is None or not self.use_condition:
            attn_out, _ = self.mha(x, x, x)
        # computing with condition changes, so that q is from x and k,v are from condition
        # in this case each "pixel" attends to all condition tokens (but in this case we have only 1 token for condition since this is just vector of embedding_size)
        # the output is the same since  Q 
        # H*W,C and K,V - 1,C
        # QK - H*W,1
        # QK * V - H*W,1 * 1,C -> H*W,C

        elif condition is not None and self.use_condition:
            # Need to project condition to the "len of token in Q"
            condition_projected = self.linear_condition(condition)
            # add one more dimension to keep the shapes as above in comments
            condition_projected = condition_projected.unsqueeze(1)
            attn_out, _ = self.mha(x, condition_projected, condition_projected)
        
        attn_out = attn_out.permute(0, 2, 1).reshape(b, c, h, w)
        return self.norm(attn_out) + x_input
    

class UNetTimed(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=100):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        
        c = 64
        self.doubleconv1 = DoubleConvLayer(in_channels, c)
        # 28x28 -> 14x14
        self.downsample1 = DownSampleLayer(c, c * 2)
        self.selfattention1 = SelfAttentionLayer(c * 2)
        
        # 14x14 -> 7x7
        self.downsample2 = DownSampleLayer(c * 2, c * 4)
        self.selfattention2 = SelfAttentionLayer(c * 4)
        
        self.middle_conv1 = DoubleConvLayer(c * 4, c * 4)
        self.middle_conv2 = nn.Sequential(
            DoubleConvLayer(c * 4, c * 4),
            DoubleConvLayer(c * 4, c * 4),
            DoubleConvLayer(c * 4, c * 4),
            DoubleConvLayer(c * 4, c * 4),
        )
        self.middle_conv3 = DoubleConvLayer(c * 4, c * 4)
        
        # 7x7 -> 14x14
        self.upsample2 = UpSampleLayer(c * 4, c * 2)
        self.selfattention5 = SelfAttentionLayer(c * 2)
        
        # 14x14 -> 28x28
        self.upsample3 = UpSampleLayer(c * 2, c)
        self.selfattention6 = SelfAttentionLayer(c)
        
        self.out = nn.Conv2d(c, out_channels, 1)
    
    def forward(self, x, t, condition = None):
        # condition just to have same interface
        time_latent = self.time_embedding(t)
        
        # In
        x1 = self.doubleconv1(x)  # 28x28

        # DownSampling
        x2 = self.downsample1(x1, time_latent)  # 14x14
        x2 = self.selfattention1(x2)
        
        x3 = self.downsample2(x2, time_latent)  # 7x7
        x3 = self.selfattention2(x3) # не берем
        
        # Bottleneck
        x3 = self.middle_conv1(x3)  # 7x7
        x3 = self.middle_conv2(x3)  # 7x7
        x3 = self.middle_conv3(x3)  # 7x7


        # Upsampling
        x = self.upsample2(x3, x2, time_latent)  # 14x14 (in last, prev 14x14, time)
        x = self.selfattention5(x)
        
        x = self.upsample3(x, x1, time_latent)  # 28x28 (in last, prev 28x28, time)
        x = self.selfattention6(x)  # 28x28

        # Out
        return self.out(x)


class UNetTimedWithVAE(nn.Module):
    def __init__(self, vae, in_channels=1, out_channels=1, time_dim=100):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.vae = vae
        c = 64
        self.doubleconv1 = DoubleConvLayer(in_channels, c)
        
        # 28x28 -> 14x14
        self.downsample1 = DownSampleLayer(c, c * 2)
        self.selfattention1 = SelfAttentionLayer(c * 2)
        self.double_conv2 = DoubleConvLayer(c * 2, c * 2)
        
        # 14x14 -> 7x7
        self.downsample2 = DownSampleLayer(c * 2, c * 4)
        self.selfattention2 = SelfAttentionLayer(c * 4)
        
        self.middle_conv1 = DoubleConvLayer(c * 4, c * 4)
        self.middle_conv2 = DoubleConvLayer(c * 4, c * 4)
        self.middle_conv3 = DoubleConvLayer(c * 4, c * 4)
        
        # 4x4 -> 7x7
        self.upsample1 = UpSampleLayer(c * 8, c * 4)
        self.selfattention4 = SelfAttentionLayer(c * 4)
        
        # 7x7 -> 14x14
        self.upsample2 = UpSampleLayer(c * 4, c * 2)
        self.selfattention5 = SelfAttentionLayer(c * 2)
        
        # 14x14 -> 28x28
        self.upsample3 = UpSampleLayer(c * 2, c)
        self.selfattention6 = SelfAttentionLayer(c)
        self.double_conv4 = DoubleConvLayer(c * 2, c * 2)

        self.double_conv_last = DoubleConvLayer(c , c )
        
        self.out = nn.Conv2d(c, out_channels, 1)
    
    def forward(self, x, t, condition = None):
        #condition just to have same interface

        time_latent = self.time_embedding(t)
        # x = self.vae.encode(x)
        # x = self.vae.reparameterize(x[0], x[1])
        # print(x.shape)
        # # x = x.view(x.shape[0], -1, 1, 1)
        # # reshape to B,1,8,8
        # x = x.view(x.shape[0], 1, 8, 8)
        # In
        x1 = self.doubleconv1(x)  # 28x28

        # DownSampling
        x2 = self.downsample1(x1, time_latent)  # 14x14
        x2 = self.selfattention1(x2)
        x2 = self.double_conv2(x2)
        
        x3 = self.downsample2(x2, time_latent)  # 7x7
        x3 = self.selfattention2(x3) # не берем
        
        # Bottleneck
        x3 = self.middle_conv1(x3)  # 7x7
        x3 = self.middle_conv2(x3)  # 7x7
        x3 = self.middle_conv3(x3)  # 7x7


        # Upsampling
        x = self.upsample2(x3, x2, time_latent)  # 14x14 (in last, prev 14x14, time)
        x = self.selfattention5(x)
        x = self.double_conv4(x2)
        

        x = self.upsample3(x, x1, time_latent)  # 28x28 (in last, prev 28x28, time)
        x = self.selfattention6(x)  # 28x28
        x = self.double_conv_last(x)
        # Out
        x = self.out(x)
        #reshape to B,64
        # Decoder
        # x = self.vae.decoder(x)
        return x
    
    def encode_vae(self,x):
        """
        x - B,1,28,28
        """
        x = self.vae.encode(x)
        x = self.vae.reparameterize(x[0], x[1])
        # x = x.view(x.shape[0], -1, 1, 1)
        # reshape to B,1,8,8
        x = x.view(x.shape[0], 1, 8, 8)
        return x

    def decode_vae(self,x):
        """
        x - B,1,8,8
        """
        x = x.view(x.shape[0], 64)
        x = self.vae.decoder(x)
        return x



class UNetTimedWithVAEConditioned(nn.Module):
    def __init__(self, vae, in_channels=1, out_channels=1, time_dim=100, condition_dim=64, use_condition=True):
        super().__init__()

        self.use_condition = use_condition
        self.time_embedding = TimeEmbedding(time_dim)
        self.condition_embedding = nn.Embedding(10, condition_dim)
        self.null_embedding = nn.Parameter(torch.randn(condition_dim))
        self.vae = vae
        c = 64
        self.doubleconv1 = DoubleConvLayer(in_channels, c)
        
        # 28x28 -> 14x14
        self.downsample1 = DownSampleLayer(c, c * 2,use_condition)
        self.selfattention1 = SelfAttentionLayer(c * 2, use_condition)
        
        # 14x14 -> 7x7
        self.downsample2 = DownSampleLayer(c * 2, c * 4,use_condition)
        self.selfattention2 = SelfAttentionLayer(c * 4, False)
        
        self.middle_conv1 = DoubleConvLayer(c * 4, c * 4)
        self.middle_conv2 = DoubleConvLayer(c * 4, c * 4)
        # self.selfattention_bottleneck = SelfAttentionLayer(c * 4, False)
        # self.crossattention_bottleneck = SelfAttentionLayer(c * 4, use_condition)
        # 7x7 -> 7x7
        self.middle_conv3 = DoubleConvLayer(c * 4, c * 4)
        
        # # 4x4 -> 7x7
        # self.upsample1 = UpSampleLayer(c * 8, c * 4)
        # self.selfattention4 = SelfAttentionLayer(c * 4, use_condition)
        
        # 7x7 -> 14x14
        self.upsample2 = UpSampleLayer(c * 4, c * 2,use_condition)
        self.selfattention5 = SelfAttentionLayer(c * 2, use_condition)
        
        # 14x14 -> 28x28
        self.upsample3 = UpSampleLayer(c * 2, c, use_condition)
        self.selfattention6 = SelfAttentionLayer(c, False)
        
        self.out = nn.Conv2d(c, out_channels, 1)

        # self.add_self_attention_encoder = SelfAttentionLayer(c*4,True)
    
    def get_condition_latent(self, condition : torch.Tensor):
        safe_condition = torch.where(condition == -1, 
                                torch.zeros_like(condition),  # replace -1 with valid index 0
                                condition)
        res = torch.where(
            condition.unsqueeze(-1) == -1,
            self.null_embedding.expand(condition.shape[0], -1),
            self.condition_embedding(safe_condition)
        )
        return res


    def forward(self, x, t, condition: torch.Tensor):

        time_latent = self.time_embedding(t)

        # will get condition latent based on id if id is -1, then will give null latent

        
        if self.use_condition:
            condition_latent = self.get_condition_latent(condition)
        else:
            condition_latent = None

        # x = self.vae.encode(x)
        # x = self.vae.reparameterize(x[0], x[1])
        # print(x.shape)
        # # x = x.view(x.shape[0], -1, 1, 1)
        # # reshape to B,1,8,8
        # x = x.view(x.shape[0], 1, 8, 8)
        # In
        x1 = self.doubleconv1(x)  # 28x28


        # DownSampling
        x2 = self.downsample1(x1, time_latent,condition_latent)  # 14x14
        x2 = self.selfattention1(x2, condition_latent)
        
        x3 = self.downsample2(x2, time_latent,condition_latent)  # 7x7
        x3 = self.selfattention2(x3, condition_latent)
        
        # Bottleneck
        x3 = self.middle_conv1(x3)  # 7x7
        x3 = self.middle_conv2(x3)  # 7x7
        # x3 = self.selfattention_bottleneck(x3, condition_latent)
        # x3 = self.crossattention_bottleneck(x3, condition_latent)
        x3 = self.middle_conv3(x3)  # 7x7


        # Upsampling
        x = self.upsample2(x3, x2, time_latent,condition_latent)  # 14x14 (in last, prev 14x14, time)
        x = self.selfattention5(x, condition_latent)
        
        x = self.upsample3(x, x1, time_latent,condition_latent)  # 28x28 (in last, prev 28x28, time)
        x = self.selfattention6(x, condition_latent)  # 28x28
        # Out
        x = self.out(x)
        # print("last",x.shape)
        #reshape to B,64
        # Decoder
        # x = self.vae.decoder(x)
        return x
    
    def encode_vae(self,x):
        """
        x - B,1,28,28
        """
        x = self.vae.encode(x)
        x = self.vae.reparameterize(x[0], x[1])
        # x = x.view(x.shape[0], -1, 1, 1)
        # reshape to B,1,8,8
        x = x.view(x.shape[0], 1, 8, 8)
        return x

    def decode_vae(self,x):
        """
        x - B,1,8,8
        """
        x = x.view(x.shape[0], 64)
        x = self.vae.decoder(x)
        return x
