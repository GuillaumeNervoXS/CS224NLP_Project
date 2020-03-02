"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layersQanet as layers
import torch
import torch.nn as nn


class QANet(nn.Module):
    """Baseline QANet model for SQuAD.

    Based on the paper:
    "QANET: COMBINING LOCAL CONVOLUTION WITH GLOBAL SELF-ATTENTION FOR
    READING COMPREHENSION"
    by Adams Wei Yu, David Dohan, Minh-Thang Luong
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained char vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob_{word,char} (float): Dropout probability.
        n_emb_enc_blocks (int): Number of encoder blocks after the embedding layer
        n_mod_enc_blocks (int): Number of encoder blocks after the attention layer
        kernel_size (int): Kernel size in the Encoder Blocks
        n_heads (int): Number of heads in the self-attention mechanisms
    """
    def __init__(self, word_vectors, char_vectors, size_char_emb ,device,hidden_size=128,
                 n_emb_enc_blocks=1,n_mod_enc_blocks=7,
                 n_conv_emb_enc=4,n_conv_mod_enc=2,
                 drop_prob_word=0.1,drop_prob_char=0.05,
                 kernel_size_emb_enc_block=7, kernel_size_mod_enc_block=7,
                 n_heads=8):
        super(QANet,self).__init__()
        self.n_emb_enc_blocks=n_emb_enc_blocks
        self.n_mod_enc_blocks=n_mod_enc_blocks
        self.n_conv_emb_enc=n_conv_emb_enc
        self.n_conv_mod_enc=n_conv_mod_enc
        self.hidden_size=hidden_size
        self.device=device
        
        self.total_layers_emb_enc=n_emb_enc_blocks*(n_conv_emb_enc+2)
        self.total_layers_mod_enc=n_mod_enc_blocks*(n_conv_mod_enc+2)
        
        self.emb = layers.EmbeddingQANet(word_vectors=word_vectors,
                                         char_vectors=char_vectors,
                                         size_char_emb=size_char_emb,
                                         hidden_size=hidden_size,
                                         drop_prob_word=drop_prob_word,
                                         drop_prob_char=drop_prob_char,
                                         out_channels=100)
        
        self.emb_enc = nn.ModuleList([layers.EncoderBlock(n_conv=n_conv_emb_enc,
                                     device=device,
                                     hidden_size=hidden_size,
                                     drop_prob=drop_prob_word,
                                     kernel_size=kernel_size_emb_enc_block,
                                     n_heads=n_heads) for _ in range(n_emb_enc_blocks)])
        
        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob_word)
        
        self.resize_layer = nn.Conv1d(4 * hidden_size, hidden_size, kernel_size=1)

        self.mod_enc = nn.ModuleList([layers.EncoderBlock(n_conv=n_conv_mod_enc,
                                     device=device,
                                     hidden_size=hidden_size,
                                     drop_prob=drop_prob_word,
                                     kernel_size=kernel_size_mod_enc_block,
                                     n_heads=n_heads) for _ in range(n_mod_enc_blocks)])

        self.out = layers.QANetOutput(hidden_size=hidden_size)
        

    def forward(self, cw_idxs,cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        #c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        #cw_idxs = cw_idxs[:, :c_len]
        #cc_idxs = cc_idxs[:, :c_len, :]
        #qw_idxs = qw_idxs[:, :q_len]
        #qc_idxs = qc_idxs[:, :q_len, :]
        #c_mask = c_mask[:, :c_len]
        #q_mask = q_mask[:, :q_len]

        c_emb=self.emb(cw_idxs,cc_idxs) # (batch_size, c_len, hidden_size)

        q_emb=self.emb(qw_idxs,qc_idxs) # (batch_size, q_len, hidden_size)
        for i,block in enumerate(self.emb_enc):
            c_emb = block(c_emb, c_mask, i*(self.n_conv_emb_enc+2)+1, self.total_layers_emb_enc)
            q_emb = block(q_emb, q_mask, i*(self.n_conv_emb_enc+2)+1, self.total_layers_emb_enc)
        att = self.att(c_emb, q_emb,c_mask, q_mask) # (batch_size, c_len, 4 * hidden_size)
        att=att.permute(0,2,1)
        att=self.resize_layer(att) # (batch_size, c_len, hidden_size)
        att=att.permute(0,2,1)
        for i,block in enumerate(self.mod_enc):
            att=block(att,c_mask,i*(self.n_conv_mod_enc+2)+1,self.total_layers_mod_enc)
        M0=att # (batch_size, c_len, hidden_size)
        for i,block in enumerate(self.mod_enc):
            att=block(att,c_mask,i*(self.n_conv_mod_enc+2)+1,self.total_layers_mod_enc)
        M1=att # (batch_size, c_len, hidden_size)
        for i,block in enumerate(self.mod_enc):
            att=block(att,c_mask,i*(self.n_conv_mod_enc+2)+1,self.total_layers_mod_enc)
        M2=att # (batch_size, c_len, hidden_size)
        out=self.out(M0,M1,M2,c_mask)
        
        return out
        
