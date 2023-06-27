#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:00:10 2018

@author: zehra
"""

#unetZehra.py
#unetZehra.py:
from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from numpy.random import normal
from math import sqrt
import pdb
import numpy as np
from torch.autograd import Variable
## install positional-encodings from https://github.com/tatp22/multidim-positional-encoding
from positional_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D,Summer,PositionalEncodingPermute1D,PositionalEncodingPermute2D

class GSMNet(nn.Module):
    def __init__(self,W=16,ImSize=192,NormalisationFirst=False,Affine=True,Activation=nn.PReLU(),Normalisation="Group", KernelSize=3):
        super(GSMNet, self).__init__()

        self.W=W 
        self.ImSize=ImSize

        self.encoder=Encoder(Activation=Activation,W=W,NormalisationFirst=NormalisationFirst)
        
            
        self.Unet_Layer6=nn.Sequential(
                ConvLayer(32*W, 16*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine),
                ConvLayer(16*W, 8*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine))



        self.Unet_Layer7=nn.Sequential(
                ConvLayer(16*W, 8*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine),
                ConvLayer(8*W, 4*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine))

        self.Unet_Layer8=nn.Sequential(
                ConvLayer(8*W, 4*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine),
                ConvLayer(4*W, 2*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine))

        self.Unet_Layer9B=nn.Sequential(
                            ConvLayer(4*W, 2*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine),
                            ConvLayer(2*W, 2*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine))

        self.Unet_Layer9=nn.Sequential(
                ConvLayer(2*W, 2*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine),
                nn.Conv2d(2*W, 1,1))        

        self.Unet_Layer6R=ConvLayer(32*W,8*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine)    

        self.Unet_Layer7R=ConvLayer(16*W,4*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine)    

        self.Unet_Layer8R=ConvLayer(8*W,2*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine)  

        self.Unet_Layer9BR=ConvLayer(4*W,2*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine) 
        
                
        self.LSTM51=nn.LSTM(8*W, 1*W,2,bidirectional=True, dropout=0.2,batch_first=True)
        self.LSTM52=nn.LSTM(8*W, 1*W,2,bidirectional=True, dropout=0.2,batch_first=True)
        self.GroupNorm=nn.GroupNorm(2,2*W)
        self.GroupNorm2=nn.GroupNorm(8,8*W)
   

        self.transformer =  nn.TransformerEncoderLayer(d_model=int(self.ImSize/16)**2, nhead=1,batch_first=True,dropout=0.2,activation= "gelu",dim_feedforward=72*4)
            


        self.Unet_Layer_ConvLSTM= ConvLSTM(nf= 8*W, in_chan=16*W,InputSize=int(self.ImSize/16))



        self.Layer1x1_5= ConvLayer(16*W, 2*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=True,stride=1,Affine=Affine)

        self.Layer1x1_6= ConvLayer(8*W, 2*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=True,stride=1,Affine=Affine)

        self.Layer1x1_7= ConvLayer(4*W, 2*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=True,stride=1,Affine=Affine)

        self.Layer1x1_8= ConvLayer(2*W, 2*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=True,stride=1,Affine=Affine)  
        
        
        self.Upsample=nn.UpsamplingBilinear2d (scale_factor=2)
        
        self._initialize_weights()
  
    def SequenceEstimation(self,xup5,xup6,xup7,xup8):
        ImSize=self.ImSize

        xup5= self.Layer1x1_5(xup5)
        xup5=F.avg_pool2d(xup5,int(ImSize/16))[:,:,:,0].permute(0,2,1)


        xup6= self.Layer1x1_6(xup6)
        xup6=F.avg_pool2d(xup6,int(ImSize/8))[:,:,:,0].permute(0,2,1)

        
        xup7= self.Layer1x1_7(xup7)
        xup7=F.avg_pool2d(xup7,int(ImSize/4))[:,:,:,0].permute(0,2,1)
       
        xup8= self.Layer1x1_8(xup8) 
        
        xup8=F.avg_pool2d(xup8,int(ImSize/2))[:,:,:,0].permute(0,2,1)


        AveragePooling2d= torch.cat((xup5,xup6,xup7,xup8),2)


        AveragePooling2d=self.GroupNorm2(AveragePooling2d.permute(0,2,1)).permute(0,2,1)

        del xup8

        torch.cuda.empty_cache()
        AveragePooling2d=AveragePooling2d.permute(1,0,2)

        b,c,w=AveragePooling2d.shape 
        Scales1= self.LSTM51(AveragePooling2d.contiguous())[0]
        Scales2= self.LSTM52(AveragePooling2d.contiguous())[0]

        LN1=Scales1-Scales2
        b,c,w=LN1.shape

        LN1=self.GroupNorm(LN1.permute(1,0,2)[:,0,:,None,None])
        
        return LN1
    

        
    def forward(self,x):


        x1,x2,x3,x4,xup5= self.encoder(x)


        b,c,w,h=xup5.shape 


        ### transformer 
        xup5=xup5.flatten(2)
        xup5=xup5.permute(1,0,2)
        b,c,h=xup5.shape
        p_enc_1d_model_sum = Summer(PositionalEncodingPermute1D(c))

        xup5=p_enc_1d_model_sum(xup5)
        xup5=self.transformer(xup5).view(b,c,w,w).permute(1,0,2,3) 

        ## convLSTM
        xup5=self.Unet_Layer_ConvLSTM(xup5[None,:,:,:,:])[0,:,:,:,:]+xup5




        b,c,w,h=xup5.shape 

        #####################3


        x4= Concatenate(x4, self.Upsample(xup5))
        x4 =self.Unet_Layer6(x4)+self.Unet_Layer6R(x4)
        
        x3= Concatenate(x3, self.Upsample(x4))
        x3 =self.Unet_Layer7(x3)+self.Unet_Layer7R(x3)


        x2= Concatenate(x2, self.Upsample(x3) )
        x2 =self.Unet_Layer8(x2)+self.Unet_Layer8R(x2)

        ###################


        LN1=self.SequenceEstimation(xup5,x4,x3,x2)

        
        del x3
        del x4
        del xup5

        torch.cuda.empty_cache()
        
        


        x2=LN1*x2


        x1= Concatenate(x1, self.Upsample(x2))

        x1=self.Unet_Layer9B(x1) +self.Unet_Layer9BR(x1)


        del x2
        
        torch.cuda.empty_cache()

        return self.Unet_Layer9(x1),LN1     

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = 0.001*torch.randn(np.shape(m.weight.data)[0],np.shape(m.weight.data)[1]) 
                m.bias.data.zero_()      


class Encoder(nn.Module):
    def __init__(self,KernelSize=3, Normalisation="Group",W=8, Activation=nn.ReLU(), NormalisationFirst=False,Affine=True):
        super(Encoder, self).__init__()
        
        
        self.Unet_Layer1=nn.Sequential(
                ConvLayer(1, 2*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine),
                ConvLayer(2*W, 2*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine))
        
        self.Unet_Layer2=nn.Sequential(
                ConvLayer( 2*W,4*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine),
                ConvLayer(4*W, 4*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=2,Affine=Affine))

        self.Unet_Layer3=nn.Sequential(
                ConvLayer( 4*W,8*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine),
                ConvLayer(8*W, 8*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=2,Affine=Affine))
        
        self.Unet_Layer4=nn.Sequential(
                ConvLayer( 8*W,16*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine),
                ConvLayer(16*W, 16*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=2,Affine=Affine))


        self.Unet_Layer5=nn.Sequential(
                ConvLayer(16*W, 32*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine),
                ConvLayer(32*W, 16*W,KernelSize, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=2,Affine=Affine))
        
        self.Unet_Layer1R=ConvLayer(1, 2*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=1,Affine=Affine)    
        
        self.Unet_Layer2R=ConvLayer(2*W,4*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=2,Affine=Affine)    
        
        self.Unet_Layer3R=ConvLayer(4*W,8*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=2,Affine=Affine)    
        
        self.Unet_Layer4R=ConvLayer(8*W,16*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=2,Affine=Affine)    
        
        self.Unet_Layer5R=ConvLayer(16*W,16*W,KernelSize=1, Normalisation=Normalisation,W=W, Activation=Activation,NormalisationFirst=NormalisationFirst,stride=2,Affine=Affine)  

        self.Dropout3=nn.Dropout2d(p=0.2)
        
        self._initialize_weights()
        
        
    def forward(self,x):
       
        
        x1= self.Unet_Layer1(x)+self.Unet_Layer1R(x)
        
        x2 = self.Unet_Layer2(x1)+self.Unet_Layer2R(x1)
        
        x3 = self.Unet_Layer3(x2)+self.Unet_Layer3R(x2)
        
        x4 = self.Unet_Layer4(x3)+self.Unet_Layer4R(x3)
        
        x5 =self.Unet_Layer5(x4)+self.Unet_Layer5R(x4)
        
        x5=self.Dropout3(x5)

        return x1,x2,x3,x4,x5        
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = 0.001*torch.randn(np.shape(m.weight.data)[0],np.shape(m.weight.data)[1]) 
                m.bias.data.zero_() 
                


                
class ConvLayer(nn.Module):
    def __init__(self,InChannelNo,OutChannelNo,KernelSize, Normalisation,W, Activation,NormalisationFirst=True,stride=1,Affine=True):
        super(ConvLayer, self).__init__()
        
        if KernelSize==3:
            self.Layer=nn.Conv2d(InChannelNo, OutChannelNo, KernelSize,padding=1,stride=stride)
           
        elif KernelSize==1:
            self.Layer=nn.Conv2d(InChannelNo, OutChannelNo, KernelSize,padding=0,stride=stride)
            
        elif KernelSize==5:
            self.Layer=nn.Conv2d(InChannelNo, OutChannelNo, KernelSize,padding=2,stride=stride)            
        elif KernelSize==7:
            self.Layer=nn.Conv2d(InChannelNo, OutChannelNo, KernelSize,padding=3,stride=stride) 
            
        self.NormalisationFirst=NormalisationFirst
        self.Activation=Activation

        if Normalisation=="Group":
            self.Normalisation=nn.GroupNorm(int(OutChannelNo/W), OutChannelNo,affine=Affine)

        elif Normalisation=="Batch":
            self.Normalisation=nn.BatchNorm2d(OutChannelNo)            
        elif Normalisation=="Instance":
            self.Normalisation=nn.InstanceNorm2d(OutChannelNo)            
            
        elif Normalisation==None:
            self.Normalisation=None
        self._initialize_weights()
        
        
    def forward(self,x,training=False):
        x=self.Layer(x)

        if self.Normalisation==None:
            x=self.Activation(x)
        else: 
            
            if self.NormalisationFirst:
                x=self.Activation(self.Normalisation(x))

            else:
                x=self.Normalisation(self.Activation(x))
        
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = 0.001*torch.randn(np.shape(m.weight.data)[0],np.shape(m.weight.data)[1]) 
                m.bias.data.zero_() 
                
                

def Concatenate(F1,F2):
    

    return torch.cat((F1,F2),1)


               
class ConvLSTMCell5(nn.Module):
    # https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf

    def __init__(self, input_dim, hidden_dim, kernel_size,InputSize=int(192/2), bias=True):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell5, self).__init__()
        self.InputSize=InputSize #int(192/2)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=8 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias,groups=2)


        self.LayerNorm= nn.Sequential(nn.LayerNorm([8*hidden_dim,self.InputSize,self.InputSize]),)
                             

            
        self.W_ci=nn.Parameter(torch.zeros(1,self.hidden_dim,self.InputSize,self.InputSize))
        

        self.W_cf=nn.Parameter(torch.zeros(1,self.hidden_dim,self.InputSize,self.InputSize))
        self.W_co=nn.Parameter(torch.zeros(1,self.hidden_dim,self.InputSize,self.InputSize))
        
        self._initialize_weights()
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv_I = self.LayerNorm(self.conv(combined))
        cc_xi, cc_xf, cc_xc, cc_xo,  cc_hi, cc_hf, cc_hc, cc_ho  = torch.split(combined_conv_I, self.hidden_dim, dim=1)

        
        i=torch.sigmoid(cc_xi+cc_hi+self.W_ci*c_cur)
        f=torch.sigmoid(cc_xf+cc_hf+self.W_cf*c_cur)
        o=torch.sigmoid(cc_xo+cc_ho+self.W_co*c_cur)
        
        
        c_next= f*c_cur+i * torch.tanh(cc_xc+cc_hc)
        h_next = o * torch.tanh(c_next)



        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)))

    def _initialize_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, sqrt(2. / n))
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = 0.001*torch.randn(np.shape(m.weight.data)[0],np.shape(m.weight.data)[1]) 
                m.bias.data.zero_()  


class ConvLSTM(nn.Module):
    def __init__(self, nf, in_chan,InputSize):
        super(ConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell5(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3), 
                                               InputSize=InputSize,
                                               bias=True)
    def autoencoder(self, x, seq_len, future_step, h_t, c_t):

        outputs1 = []
        outputs2 = []
        h_t_init=h_t
        c_t_init=c_t
        # encoder
        seq_len=x.shape[1]
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t,:, :, :],cur_state=[h_t, c_t]) 
            outputs1+= [h_t]
        
        h_t=h_t_init
        c_t=c_t_init
        
        for t in range(seq_len-1,-1,-1):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t,:, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            outputs2+= [h_t]
        
         
        outputs1 = torch.stack(outputs1, 1)
        outputs2 = torch.stack(outputs2, 1)
        outputs=torch.cat((outputs1,outputs2),2)
        return outputs

    def forward(self, x, future_seq=24, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()
        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t)

        return outputs
    
