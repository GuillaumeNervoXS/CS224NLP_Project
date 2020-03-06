"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import masked_softmax, PosEncoder
import numpy as np


class Embedding(nn.Module):
    """Embedding layer used by QANet, with a character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    
    def __init__(self, word_vectors, char_vectors, size_char_emb, hidden_size, drop_prob_word=0.1, drop_prob_char=0.05, out_channels=100):
        super(Embedding, self).__init__()
        self.drop_prob_word = drop_prob_word
        self.drop_prob_char = drop_prob_char
        self.word_emb_dim   = word_vectors.size(1)
        self.char_emb_dim   = size_char_emb
        self.size_char_vocab = char_vectors.size(0)
        self.out_channels   = out_channels
        
        self.embed_word = nn.Embedding.from_pretrained(word_vectors,freeze=True)
        self.embed_char = nn.Embedding(num_embeddings=self.size_char_vocab, embedding_dim=size_char_emb)
        
        self.cnn=CNN(self.char_emb_dim, output_channels=self.out_channels)
        
        self.proj = nn.Linear(self.word_emb_dim+self.out_channels, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x_word , x_char):
        emb_w = self.embed_word(x_word)   # (batch_size, seq_len, word_emb_dim)
        emb_w = F.dropout(emb_w, self.drop_prob_word, self.training)
        
        emb_c = self.embed_char(x_char)   # (batch_size, seq_len, word_len ,char_emb_dim)
        
        #use cnn to have character level representation
        batch_size, seq_len, word_len, _ = emb_c.shape
        view_shape = (batch_size * seq_len, word_len, self.char_emb_dim)
        emb_c      = emb_c.view(view_shape).transpose(1,2)
        emb_c = F.dropout(emb_c, self.drop_prob_char, self.training)
        emb_c_conv = self.cnn(emb_c)
        emb_c_conv = emb_c_conv.view(batch_size, seq_len, self.out_channels)
        
        #concatenate both embedding with the righ dim
        emb = torch.cat((emb_w,emb_c_conv),dim=-1)
        

        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb
       

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x



class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        @hidden_size (int): Size of hidden activations.
        @drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight= nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)
        
        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class CNN(nn.Module):
    """
        This class implements the 1D-Convolution components of the CharEmbedding model
    """
    
    def __init__(self, emb_size, output_channels, kernel_size=5, padding=1):
        """
            Init the Convolution Network
            @param emb_size: The size of the char embedding
            @param kernel_size: The number of char embedding (letter) we want to consider when we
                                aply the convolution filter
            @param output_channels: The number of filter apply to each pass
        """
        
        super(CNN, self).__init__()
        
        self.conv1D=nn.Conv1d(in_channels=emb_size,out_channels=output_channels,
                              kernel_size=kernel_size,padding=padding,
                              bias=True)
    def forward(self,inpt):
        """
            Applies a 1d-convolutional filter to every window of size kernel_size to an input tensor of
            size (sentence_length*BATCH_SIZE,  e_char, word_length) to get a new tensor of size
            (sentence_length*BATCH_SIZE, output_channels, word_length-kernel_size+3)
            Then we apply a max_pool to have identical lenght word embeddings.
            
            @param inpt: Tensor of shape (sentence_length*BATCH_SIZE, e_char, word_length) representing
                        char embedding (of size e_char) for each char in each word in each batch in each sentences
                        
            @param x_conv_out: Tensor of shape (sentence_length*BATCH_SIZE, output_channels)
                                which is the current word embedding before highway (note that the dimension
                                of all words are equal to output_channels whichever size of the word).
        """
                
        x_conv=F.relu(self.conv1D(inpt))
        x_conv_out,_=torch.max(x_conv,-1)
        
        return x_conv_out


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolutions used in the QANet paper
    Info on this type of convolutions here: 
    https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
    Args:
        in_channels (int):  # of in_channels of the convolution
        out_channels (int): # of in_channels of the convolution
    """
    def __init__(self, in_channels, out_channels, k, bias=True):
        super(DepthwiseSeparableConv,self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=k, groups=in_channels, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=bias)
        
    def forward(self, x):
        x=x.permute(0,2,1)
        x=F.relu(self.pointwise_conv(self.depthwise_conv(x)))
        return x.permute(0,2,1)
  
    
class SelfAttention(nn.Module):
    """Self-attention mechanism used in the Encoder Block
    Multi-head attention mechanism defined in the Transformer paper which,
    for each position in the input called the query, computes a weighted sum of all
    positions, or keys, in the input based on the similarity between the query 
    and key as measured by the dot product.
    
    Args:
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self,hidden_size=128,n_heads=8,divisor_dim_kqv=8,drop_prob=0.1):
        super(SelfAttention,self).__init__()
        self.hidden_size=hidden_size
        self.nbr_head_attention = n_heads
        self.dim_query_key_val=hidden_size//divisor_dim_kqv
        
        self.fc_queries=nn.ModuleList([nn.Linear(hidden_size,self.dim_query_key_val) \
                                       for _ in range(self.nbr_head_attention)])
        self.fc_keys=nn.ModuleList([nn.Linear(hidden_size,self.dim_query_key_val) \
                                    for _ in range(self.nbr_head_attention)])
        self.fc_values=nn.ModuleList([nn.Linear(hidden_size,self.dim_query_key_val) \
                                      for _ in range(self.nbr_head_attention)])
        
        self.dropout  =nn.Dropout(p=drop_prob)
        #we use the same projection as the previous step give us some kind of language 
        #and therefore we only retranscript in a smaller dimension so should be the same process at
        #every step. We can discuss about that
        self.proj_multihead=nn.Linear(self.nbr_head_attention*self.dim_query_key_val,hidden_size)
        
    def forward(self, x, mask):        
        batch_size,seq_len,hidden_size=x.size()
        mask = mask.unsqueeze(-1).expand(x.shape[0], x.shape[1], x.shape[1]).permute(0,2,1)
        #mask= mask.reshape(1,batch_size,seq_len,1)

        #we compute the query, key,value for each self attention
        queries=[self.fc_queries[i](x) for i in range(self.nbr_head_attention)] #list of size nbr_head_attention each of size (batch_size,seq_len,self.dim_query_key_val)
        keys   =[self.fc_keys[i](x) for i in range(self.nbr_head_attention)]       #list of size nbr_head_attention each of size (batch_size,seq_len,self.dim_query_key_val)
        values =[self.fc_values[i](x) for i in range(self.nbr_head_attention)]   #list of size nbr_head_attention each of size (batch_size,seq_len,self.dim_query_key_val)
        
        #we concatenate each query,key,values for efficiency
        queries=torch.stack(queries,dim=0) #(nbr_head,batch_size,seq_len,self.dim_query_key_val)
        keys   =torch.stack(keys,dim=0)    #(nbr_head,batch_size,seq_len,self.dim_query_key_val)
        values =torch.stack(values,dim=0)  #(nbr_head,batch_size,seq_len,self.dim_query_key_val)
        

        #we directly reshape queries and keys to use bmm
        queries=queries.reshape(self.nbr_head_attention*batch_size,seq_len,self.dim_query_key_val)
        keys   =keys.reshape(self.nbr_head_attention*batch_size,seq_len,self.dim_query_key_val)
        values =values.reshape(self.nbr_head_attention*batch_size,seq_len,self.dim_query_key_val)


        #we then compute scores for each combination of query,key 
        scores=torch.bmm(queries,keys.transpose(1,2))/np.sqrt(self.dim_query_key_val) #(nbr_head*batch_size,seq_len,seq_len)
        #we need to reshape the mask, I am not sure about this line maybe just repeat 
        #the mask instead of using broadcasting.
        scores=scores.reshape(self.nbr_head_attention,batch_size,seq_len,seq_len)
        scores_softmax=masked_softmax(scores,mask,dim=3) #(nbr_head,batch_size,seq_len,seq_len)
        scores_softmax=scores_softmax.reshape(self.nbr_head_attention*batch_size,seq_len,seq_len)
        
        scores_softmax_drop=self.dropout(scores_softmax)
        
        #compute the waighted average of the values thanks to the scores
        self_attented_values= torch.bmm(scores_softmax_drop,values) #(nbr_head*batch_size,seq_len,self.dim_query_key_val)
        
        #we then concatenate the multi attention along the last dim to apply the fc to get an output of the same
        #size as the input
        self_attented_values=self_attented_values.reshape(self.nbr_head_attention,
                                                          batch_size,seq_len,self.dim_query_key_val).permute(1,2,3,0)
        self_attented_values=self_attented_values.reshape(batch_size,seq_len,
                                                          self.dim_query_key_val*self.nbr_head_attention)
        
        #projection to smaller dim to have a single vector for each dimension
        x=self.proj_multihead(self_attented_values)
        
        return x


class EncoderBlock(nn.Module):
    """ Encoder Block of the QANet model
    Args: 
        n_conv (int): Number of convolutions in the block
        kernel_size (int) (default=7)
        padding(int) (default=3)
        hidden_size (int) (default=128)
        n_heads (int) (default=8)
    """
    def __init__(self, n_conv, hidden_size,drop_prob=0.1, kernel_size=7, n_heads=8, divisor_dim_kqv=8):
        super(EncoderBlock,self).__init__()
        self.n_conv = n_conv
        self.hidden_size = hidden_size
        self.drop_prob=drop_prob
        
        self.conv = nn.ModuleList([DepthwiseSeparableConv(in_channels=hidden_size, out_channels=hidden_size, 
                                                          k=kernel_size) for _ in range(n_conv)])
        self.self_att = SelfAttention(hidden_size=hidden_size,n_heads=n_heads, divisor_dim_kqv=divisor_dim_kqv)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_conv)])
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.ffl = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
    
    def layer_dropout(self, inputs, residual, dropout):
        if self.training:
            if torch.rand(1) > dropout:
                outputs = F.dropout(inputs, p=self.drop_prob, training=self.training)
                return outputs + residual
            else:
                return residual
        else:
            return inputs + residual  
        
    def forward(self, x, mask, start_index, total_layers):
        """
        x has shape (batch_size, seq_len, hidden_size)
        """
        #total_layers = (self.conv_num+1)*n_blocks
        output = PosEncoder(x)
        
        #Convolutional Layers
        for i in range(self.n_conv):
            residual = output
            output=self.layer_norm[i](output)
            if (i) % 2 == 0:
                output = F.dropout(output, p=self.drop_prob, training=self.training)
            output = self.conv[i](output) # (batch_size, seq_len, hidden_size)
            #Layer Dropout
            output = self.layer_dropout(output, residual, self.drop_prob*start_index/total_layers)
            start_index += 1
        
        #Self-Attention
        residual = output
        output = self.norm_1(output)    
        output = F.dropout(output, p=self.drop_prob, training=self.training)
        # (batch_size, hidden_size, seq_len)
        output = self.self_att(output, mask)
        output = self.layer_dropout(output, residual, self.drop_prob*start_index/total_layers)
        start_index+= 1
        
        #Fully Connected Layer
        residual = output
        output = self.norm_2(output)
        output = F.dropout(output, p=self.drop_prob, training=self.training)
        output=output.permute(0,2,1)
        output = self.ffl(output)
        output=output.permute(0,2,1)
        output = self.layer_dropout(output, residual, self.drop_prob*start_index/total_layers)
        return output

class LayerOutputStart(nn.Module):
    """
    Class which represent the start_index branch of the QANet architecture
    
    
    It consist of taking two matrix as input (which are obtained through encoder 
    block) of size (batch_size,seq_len,hidden_size), concatenate them along the 
    last axis and then use Linear layer and softmax to get log probabilities.
    
    Args:
        @hidden_size (int): the hidden size (last dim) of the input matrix
    
    """
    
    def __init__(self,hidden_size,drop_prob):
        super(LayerOutputStart,self).__init__()
        
        self.fc=nn.Linear(hidden_size,1)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self,M0,M2,c_mask):
        
        M         = torch.cat((M0,M2),dim=-1)
        scores    = self.fc(M).squeeze()
        log_probs = masked_softmax(logits=scores,mask=c_mask, log_softmax=True)
        
        return log_probs

class LayerOutputEnd(nn.Module):
    """
    Class which represent the stop_index branch of the QANet architecture
    
    
    It consist of taking two matrix as input (which are obtained through encoder 
    block) of size (batch_size,seq_len,hidden_size), concatenate them along the 
    last axis and then use Linear layer and softmax to get log probabilities.
    
    Args:
        @hidden_size (int): the hidden size (last dim) of the input matrix
    
    """
    
    def __init__(self,hidden_size,drop_prob):
        super(LayerOutputEnd,self).__init__()
        
        self.fc=nn.Linear(hidden_size,1)
        self.fc_prob=nn.Linear(2,1)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self,M1,M2,prob_start,c_mask):
        
        M         = torch.cat((M1,M2),dim=-1)
        scores_end= self.fc(M)
        scores_end= self.fc_prob(torch.cat((scores_end,prob_start.unsqueeze(-1)), dim=-1)).squeeze()
        log_probs = masked_softmax(logits=scores_end,mask=c_mask, log_softmax=True)
        
        return log_probs

class QANetOutput(nn.Module):
    """Output Layer of the QANet model
    
    """
    def __init__(self, hidden_size=128):
        super(QANetOutput, self).__init__()

        self.hidden_size = hidden_size

        self.W1 = nn.Linear(2*self.hidden_size, 1)
        self.W2 = nn.Linear(2*self.hidden_size, 1)

        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, M0, M1, M2,mask):

        p1 = self.W1(torch.cat((M0,M1), -1)).squeeze() # (batch_size, c_len)
        p2 = self.W2(torch.cat((M0,M2), -1)).squeeze() # (batch_size, c_len)
        log_p1 = masked_softmax(p1, mask,log_softmax=True)
        log_p2 = masked_softmax(p2, mask,log_softmax=True)
        return log_p1, log_p2
