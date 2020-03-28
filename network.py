# -*- encoding: utf-8 -*-
#'''
#@file_name    :network.py
#@description    :
#@time    :2020/02/13 12:47:27
#@author    :Cindy, xd Zhang 
#@version   :0.1
#'''
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sublayers import *
from data_loader import MAX_LENGTH
from torch.nn.utils.rnn import pad_sequence
Global_device="cpu"
def get_attn_pad_mask(seq_q, seq_k):
    #[B,LQ]  B,LQ,E * B,E,LK->B,LQ,LK 
    #[B,LK]
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k

def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask

class Decoder_layer(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, Embeddingsize,n_head,d_k,d_v,d_hidden,dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_head, Embeddingsize, d_k, d_v, dropout)
        self.enc_dec_attention = MultiHeadAttention(n_head, Embeddingsize, d_k, d_v, dropout)
        self.FForward = PositionwiseFeedForward(Embeddingsize, d_hidden, dropout=dropout)
    def forward(self, dec_input,self_attn_mask,enc_out,enc_dec_mask):
        #X=B,L
        # slf_attn_mask=padding_mask(Embedding_x)
        outputDec, attn=self.self_attention(dec_input,dec_input,dec_input,self_attn_mask)
        output, attn=self.self_attention(dec_input,enc_out,enc_out,enc_dec_mask)
        output=self.FForward(output)
        return output
class Encoder_layer(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, Embeddingsize,n_head,d_k,d_v,d_hidden,dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_head, Embeddingsize, d_k, d_v, dropout)
        self.FForward = PositionwiseFeedForward(Embeddingsize, d_hidden, dropout=dropout)
    def forward(self, x,slf_attn_mask):
        #X=B,L
        # slf_attn_mask=padding_mask(Embedding_x)
        output, attn=self.self_attention(x,x,x,slf_attn_mask)
        output=self.FForward(output)
        return output
class TransformerEncoder(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self,config,voc_Size,embedding_layer):
        super().__init__()
        self.char_embedding=embedding_layer
        self.Positional_Encoding=PositionalEncoding(d_hid=config.embedding_size, n_position=MAX_LENGTH)
        self.layerstack=[ Encoder_layer(config.embedding_size,config.n_head,config.d_k,config.d_v,config.d_hidden,config.dropout) for _ in range(config.n_layers)  ]

    def forward(self, x):
        #X=B,L
        slf_attn_mask=padding_mask(x)
        output=self.Positional_Encoding(self.char_embedding(x))
        for layer in self.layerstack:
            output = layer(output,slf_attn_mask=slf_attn_mask)
        return output

class TransformerDecoder(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self,config,voc_Size,embedding_layer):
        super().__init__()
        self.Positional_Encoding=PositionalEncoding(d_hid=config.embedding_size, n_position=MAX_LENGTH)
        self.layerstack=[ Decoder_layer(config.embedding_size,config.n_head,config.d_k,config.d_v,config.d_hidden,config.dropout) for _ in range(config.n_layers)  ]
        self.char_embedding=embedding_layer
    def forward(self, dec_input,enc_output,enc_input):
        #X=B,L
        slf_attn_mask=padding_mask(dec_input)
        sq_mask=sequence_mask(dec_input)

        slf_attn_mask = (torch.gt((slf_attn_mask.float() + sq_mask.float()), 0)).float()
        enc_dec_mask=get_attn_pad_mask(dec_input,enc_input)

        output=self.Positional_Encoding(self.char_embedding(dec_input))
        

        for layer in self.layerstack:
            output = layer(output,self_attn_mask=slf_attn_mask,enc_out=enc_output,enc_dec_mask=enc_dec_mask)
        return output
class Transformer(nn.Module):
    def __init__(self, config,voc_Size,criterion):
        super().__init__() 
        self.config=config
        self.char_embedding= Embeddings(voc_Size,config.embedding_size)
        self.encoder=TransformerEncoder(config,voc_Size,self.char_embedding)
        self.decoder=TransformerDecoder(config,voc_Size,self.char_embedding)
        self.criterion=criterion
        self.tgt_proj=nn.Linear(config.embedding_size, voc_Size, bias=False)
        self.final_softmax = nn.Softmax(dim=2)
    def call(self,Q,A):
        enc_output=self.encoder(Q)
        dec_input=A[:,:-1]
        dec_target=A[:,1:]
        dec_out=self.decoder(dec_input,enc_output,Q)
        dec_logits = self.final_softmax(self.tgt_proj(dec_out)) 
        loss=self.criterion(dec_logits.contiguous().view(dec_logits.size(0)*dec_logits.size(1),-1),dec_target.contiguous().view(-1))
        return loss
    def train(self,train_loader,optimizer):
        start_epoch=0
        stage_total_loss=0
        for epoch in range(start_epoch,self.config.end_epoch):
            for batch_idx,batch in enumerate(train_loader):
                optimizer.zero_grad()
                batchQ,batchA=batch['Q'],batch['A']
                batchQ = pad_sequence(batchQ,batch_first=True, padding_value=0).to(Global_device)
                batchA = pad_sequence(batchA,batch_first=True, padding_value=0).to(Global_device)
                loss=self.call(batchQ,batchA)
                loss.backward()
                optimizer.step_and_update_lr()
                stage_total_loss+=loss.cpu().item() 
                if batch_idx % self.config.log_steps == 0:
                    print_loss_avg = (stage_total_loss / self.config.log_steps)
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttime: {}'.format(
                        epoch, batch_idx , len(train_loader),
                        100. * batch_idx / len(train_loader), print_loss_avg, time.asctime(time.localtime(time.time())) ))
                    with open(self.config.logfile_path,'a') as f:
                        template=' Train Epoch: {} [{}/{}]\tLoss: {:.6f}\ttime: {}\n'
                        str=template.format(epoch,batch_idx , len(train_loader),print_loss_avg,\
                            time.asctime(time.localtime(time.time())))
                        f.write(str)
                    stage_total_loss=0
