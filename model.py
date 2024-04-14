import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocabularyulary_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocabularyulary_size = vocabularyulary_size
        # mapping between a number and vector of size d_model. 
        # the vector is learned by the model.
        self.embedding = nn.Embedding(vocabularyulary_size, d_model)


    def forward(self, x):
        # in the paper they say that weights of the embedding layer are 
        # multiplied by the sqrt(model_dimension)
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        # matrix of shape(seq_length, d_model)
        pe = torch.zero(seq_length, d_model)
        # vector of shape(seq_length, 1)
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        
        # apply sin to even positions and cosine to odd positions
        pe[:, 0::2] = torch.sin(pos * denominator)
        pe[:, 1::2] = torch.cos(pos * denominator)

        # add the batch dimension to this tensor so that we can apply it 
        # to the whole batch of sentences
        # becomes a tensor of shape(1, seq_length, d_model)
        pe = pe.unsqueeze(0)

        # this register_buffer allows to save the tensor to the model's buffer
        # model's buffer is used to keep tensors that we want to keep inside
        # the model, not as a learned parameter, but it's saved in the file 
        # when the model is saved along the model's state
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # usually means cancels the dimension to which is applied, but we want to keep it
        std = x.std(dim=1, keepdim=True)
        # apply the formula
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1 (bias=True so it's defining a bias matrix for us)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        # d_model has to be divisible by h because d_v = d_k = d_model/h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h     # dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # Wo should be (h*d_v, d_model), but h*d_v = d_model
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        # (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # to ensure some words don't interact with other words, we mask them
        # we mask by putting something very small (like -inf) in places where 
        # interactions between words we want to stop from interacting will be 
        # present, i.e., on their spots in the matrix. This makes sure that 
        # softmax outputs very close to 0 (or 0) for the interaction between 
        # masked words.
        query = self.w_q(q) # Q' from the slides
        key = self.w_k(k)   # K'
        value = self.w_v(v) #V'

        # splitting matrices on the embedding dimension into h smaller matrices to provide to the head
        # we transpose because we want h to be the 2nd dimension (not 3rd):
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,
                                                                     key,
                                                                     value,
                                                                     mask,
                                                                     self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            # in the paper they first do norm then sublayer, but there are many
            # implementations where it works in this way as well
            return x + self.dropout(sublayer(self.norm(x)))
        

class EncoderBlock(nn.Module):

    def __init__(self, features: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, source_mask):
        # source_mask is the mask that we want to apply to the input of the
        # encoder. We want to hide the interaction of padding words with other 
        # words in the input.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, features: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, source_mask, target_mask):
        # target mask applied to the decoder
        x = self.residual_connections[0](x,
                                         lambda x: self.self_attention_block(x,
                                                                             x,
                                                                             x,
                                                                             target_mask))
        # source mask applied to the encoder
        x = self.residual_connections[1](x,
                                         lambda x: self.cross_attention_block(x,
                                                                              encoder_output,
                                                                              encoder_output,
                                                                              source_mask))
        
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)
    


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocabularyulary_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocabularyulary_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocabularyulary_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    

class Transformer(nn.Module):
    """
        We build encode, decode and project and we will apply them in succession.
        We build them separately (instead of just defining one forward) because
        because during inferencing we can reuse the output of the encoder (we 
        don't have to calculate it every time) and these outputs separately are
        also useful for visualizing attention.
    """
    def __init__(self, encoder: Encoder,
                 decoder: Decoder,
                 source_embed: InputEmbeddings,
                 target_embed: InputEmbeddings,
                 source_pos: PositionalEncoding,
                 target_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, source, source_mask):
        # (batch, seq_len, d_model)
        source = self.source_embed(source)
        source = self.source_pos(source)
        return self.encoder(source, source_mask)
    
    def decode(self,
               encoder_output: torch.Tensor,
               source_mask: torch.Tensor,
               target: torch.Tensor,
               target_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)
    
    def project(self, x):
        # (batch, seq_len, vocabulary_size)
        return self.projection_layer(x)



def build_transformer(source_vocabulary_size: int,
                      target_vocabulary_size: int,
                      source_seq_len: int,
                      target_seq_len: int,
                      d_model: int=512,
                      N: int=6,
                      h: int=8,
                      dropout: float=0.1,
                      d_ff: int=2048) -> Transformer:
    """
        This function build the transformer given all the hyperparameters and
        also initialize the parameters with some initial values
    """
    # create the embedding layers
    source_embed = InputEmbeddings(d_model, source_vocabulary_size)
    target_embed = InputEmbeddings(d_model, target_vocabulary_size)

    # create the positional encoding layers. There is no actual need to create
    # two Positional encoding layers (only one works just fine), but it's done
    # like this now for clarity reasons.
    source_pos = PositionalEncoding(d_model, source_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)
    
    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,
                                                               h,
                                                               dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model,
                                     encoder_self_attention_block,
                                     feed_forward_block,
                                     dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,
                                                               h,
                                                               dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,
                                                                h,
                                                                dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model,
                                     decoder_self_attention_block,
                                     decoder_cross_attention_block,
                                     feed_forward_block,
                                     dropout)
        decoder_blocks.append(decoder_block)
    
    # create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # create the projection layer
    projection_layer = ProjectionLayer(d_model, target_vocabulary_size)
    
    # create the transformer
    transformer = Transformer(encoder,
                              decoder,
                              source_embed,
                              target_embed,
                              source_pos,
                              target_pos,
                              projection_layer)
    
    # initialize the parameters so the model doesn't start from completely
    # random parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer