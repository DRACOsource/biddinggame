# -*- coding: utf-8 -*-
import torch
from torch import nn,optim,tensor
from torch.nn import functional as F
from ..config.config import (PRICE_MODEL_PARAMS as pmp,DEVICE)

class EncoderRNN(nn.Module):
    max_length = pmp.batch_size
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_tensor, hidden):
        embedded = F.relu(self.embedding(input_tensor)).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    

class DecoderRNN(nn.Module):
    dropout_p = 0.1
    max_length = pmp.batch_size
    def __init__(self, output_size, hidden_size=128):
        super().__init__()
        
        self.embedding = nn.Linear(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, decoder_input, hidden, encoder_outputs):
        embedded = F.relu(self.embedding(decoder_input)).view(1,1,-1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = nn.Sigmoid()(self.out(output[0]))
        return output, hidden, attn_weights


class Attention(nn.Module):
    teacher_forcing_ratio = 0.5
    epoch = 5
    criterion = nn.MSELoss()
    def __init__(self,unique_id,input_size,output_size,maxReward):
        super().__init__()
        self.unique_id = unique_id + '_attentionMdl'
        self.output_size = output_size
        if maxReward is None:
            self.maxReward = pmp.maxExtReward
        else:
            self.maxReward = maxReward
        
        self.encoder = EncoderRNN(input_size)
        self.decoder = DecoderRNN(output_size,self.encoder.hidden_size)
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.avg_reward = 0
        self.trainingdata = None
    
    def setOptim(self,lr=0.01):
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(),lr=lr)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(),lr=lr)        
    
    def initHidden(self,hidden_size): 
        return torch.zeros(1, 1, hidden_size, device=DEVICE).float()

    def train(self,input_tensor,target_tensor,end_value=0):
        if end_value > self.maxReward:
            self.maxReward = end_value
        end_value = end_value / self.maxReward
        encoder_hidden = self.initHidden(self.encoder.hidden_size)
        sequence_length = input_tensor.size()[0] # input/output sequence length
        encoder_outputs = torch.zeros(self.encoder.max_length, 
                                      self.encoder.hidden_size, device=DEVICE)
        
        for ei in range(sequence_length):
            encoder_output, encoder_hidden = self.encoder(
                                             input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0] # first (virtual) input
    
        decoder_inputs = torch.cat([
                        target_tensor.view(-1,1).float(),
                        tensor([end_value],device=DEVICE).view(-1,1).float()])
        decoder_hidden = encoder_hidden
        
        loss = 0
        for di in range(sequence_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_inputs[di], decoder_hidden, encoder_outputs)
        # only predict the last output
        loss += self.criterion(decoder_output.view(-1,1),
                               decoder_inputs[di+1].view(-1,1))

        attention_loss = loss.detach()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
    
        return attention_loss
    
    def inference(self,input_tensor,target_tensor,end_value=0.0,target=None):
        # if target=='maximize', use prediction as additional factor;
        # if target=='minimize', use -prediction as additional factor;
        # if target is None, predicted value is not used in weight vector        
        maxlength = self.encoder.max_length
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            encoder_hidden = self.initHidden(self.encoder.hidden_size)
            sequence_length = input_tensor.size()[0] # input/output sequence length
            encoder_outputs = torch.zeros(maxlength, 
                                      self.encoder.hidden_size, device=DEVICE)
    
            for ei in range(sequence_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]
    
            decoder_inputs = torch.cat(
                    [tensor([-1.],device=DEVICE).view(-1,1),
                     target_tensor.view(-1,1),
                     tensor([end_value],device=DEVICE).view(-1,1)]).float()
            decoder_hidden = encoder_hidden
    
            for di in range(maxlength):
                decoder_output,decoder_hidden,decoder_attention = self.decoder(
                    decoder_inputs[di], decoder_hidden, encoder_outputs)
                
            weightVec = decoder_attention.data.squeeze()
            if target=='maximize':
                weightVec = weightVec * decoder_output[0]
            elif target=='minimize':
                weightVec = weightVec * decoder_output[0] * (-1)
            return weightVec
    