
import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np

import random
import math
import time

from pytorchtools import count_parameters, init_weights
from pytorchtools import CustomLoss, CustomLoss_perCase, CustomLoss_perYear



# ==================================================================
# FC_RNN_Model() Code origin:
#
# https://github.com/SheezaShabbir/Time-series-Analysis-using-LSTM-RNN-and-GRU/blob/main/Pytorch_LSTMs%2CRNN%2CGRU_for_time_series_data.ipynb
#
# ==================================================================


class FC_RNN_Model(nn.Module):
	"""FC_RNN_Model class extends nn.Module class and works as a constructor for GRUs.

	   FC_RNN_Model class initiates a GRU module based on PyTorch's nn.Module class.
	   It has only two methods, namely init() and forward(). While the init()
	   method initiates the model with the given input parameters, the forward()
	   method defines how the forward propagation needs to be calculated.
	   Since PyTorch automatically defines back propagation, there is no need
	   to define back propagation method.

	   Attributes:
		   hid_dim (int): The number of nodes in each layer
		   n_layers (str): The number of layers in the network
		   gru (nn.GRU): The GRU model constructed with the input parameters.
		   fc (nn.Linear): The fully connected layer to convert the final state
						   of GRUs to our desired output shape.

	"""
	def __init__(self, rnnParams):
	#def __init__(self, input_dim, hid_dim, n_layers, output_dim, dropout, rnn_type='GRU'):
		"""The __init__ method that initiates a GRU instance.

		Args (in dict: rnnParams)
			input_dim (int): The number of nodes in the input layer
			hid_dim (int): The number of nodes in each layer
			n_layers (int): The number of layers in the network
			output_dim (int): The number of nodes in the output layer
			dropout_rnn (float): The probability of nodes being dropped out (rnn)
			rnn_type (string): 'GRU' or 'LSTM'
			
			inp_dim_fc (int): Number of fully connected input block's inputs
			fc_in_sizes (list of int): The fully connected block layer sizes 
										(the last layer length must be equal to 'hid_dim_rnn'.
			dropout_fc (float): The dropout probability of the fully connected input block

		"""
		super(FC_RNN_Model, self).__init__()

		# Fully connected block parameters:
		self.inp_dim_fc = rnnParams['inp_dim_fc']
		self.fc_in_sizes = rnnParams['fc_in_sizes']
		self.dropout_fc = rnnParams['dropout_fc']

		# RNN block parameters:
		self.input_dim = rnnParams['input_dim_rnn']
		self.hid_dim = rnnParams['hid_dim_rnn']
		self.n_layers = rnnParams['n_layers_rnn']
		self.output_dim = rnnParams['output_dim']
		self.dropout_rnn = rnnParams['dropout_rnn']
		self.rnn_type = rnnParams['rnn_type']
		self.n_layers_fc2h0 = rnnParams['n_layers_fc2h0']
		
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout=self.dropout_rnn, batch_first=True)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(self.input_dim, self.hid_dim, self.n_layers, dropout=self.dropout_rnn, batch_first=True)
			
		self.fc_Sequential = FC_Sequential(self.inp_dim_fc, self.fc_in_sizes, self.dropout_fc)

		# GRU layers
		#self.gru = nn.GRU(input_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)

		# Fully connected output layer:
		self.fc_out = nn.Linear(self.hid_dim, self.output_dim)

	def forward(self, inp_seq, inp_fc):
		#print("FC_RNN_Model: inp_seq.shape = ", inp_seq.shape)
		#print("FC_RNN_Model: inp_fc.shape = ", inp_fc.shape)
		
		"""The forward method takes input tensor 'inp_seq' and does forward propagation

		Args:
			inp_seq (torch.Tensor): The input tensor to the RNN block; shape (batch size, sequence length, input_dim)
			
			inp_fc (torch.Tensor): The inputs to the fully connected block; shape (batch size, inp_dim_fc)

		Returns:
			outputs (torch.Tensor): The output tensor of the shape (batch_size, seq_len, output_dim)

		"""
		# Initializing hidden state for first input the output from the fully
		# connected network (inputting static forest variable data):
		# H0 dimensions shall be: [n_layers,batch_size,hid_dim)]
		#
		# The input parameter 'n_layers_fc2h0' controls to how many RNN
		# layers the fully connected module's outputs will be connected
		# (into h0 and c0 also in case of LSTM). The remainding layers h0
		# (& c0) will be initialized to zero.
		
		h0 = torch.zeros(self.n_layers, inp_seq.size(0), self.hid_dim, device=inp_seq.device)
		if self.rnn_type == 'LSTM':
			c0 = torch.zeros(self.n_layers, inp_seq.size(0), self.hid_dim, device=inp_seq.device)
		
		for ii in range(min(self.n_layers, self.n_layers_fc2h0)):
			h0[ii,:,:] = self.fc_Sequential(inp_fc)
			#h0[ii,:,:] = self.fc_Sequential(inp_fc).unsqueeze(1)
			
			# If RNN = LSTM, initialize cell state with FC block outputs also:
			if self.rnn_type == 'LSTM':
				c0[ii,:,:] = self.fc_Sequential(inp_fc)
				#c0[ii,:,:] = self.fc_Sequential(inp_fc).unsqueeze(1)
		
		# Initializing hidden state for first input with zeros
		#h0 = torch.zeros(self.n_layers, inp_seq.size(0), self.hid_dim,device=inp_seq.device).requires_grad_()
		
		# Forward propagation by passing in the input and hidden state into the model
		# The outputs should be in the shape: [batch_size, seq_length, hid_size]
		if self.rnn_type == 'GRU':
			outputs, _ = self.rnn(inp_seq, h0)
		else:
			outputs, (_,_) = self.rnn(inp_seq, (h0, c0))
		
		#if self.rnn_type == 'GRU':
		#	outputs, _ = self.rnn(inp_seq, h0.detach())
		#else:
		#	outputs, (_,_) = self.rnn(inp_seq, (h0.detach(), c0.detach()))
			
		#print("FC_RNN_Model: outputs.shape = ", outputs.shape)
		
		#outputs, _ = self.gru(inp_seq, h0.detach())

		# Reshaping the outputs in the shape of (batch_size, seq_length, hid_size)
		# so that it can fit into the fully connected layer
		#outputs = outputs[:, -1, :]

		# Convert the final state to our desired output shape (batch_size, seq_len, output_dim)
		outputs = self.fc_out(outputs)

		return outputs

# ==================================================================
# FC_RNN_FC_Model() Code origin:
#
# https://github.com/SheezaShabbir/Time-series-Analysis-using-LSTM-RNN-and-GRU/blob/main/Pytorch_LSTMs%2CRNN%2CGRU_for_time_series_data.ipynb
#
# ==================================================================
#
# INCOMPLETE; NOT USED!

class FC_RNN_FC_Model(nn.Module):
	"""RNNModel class extends nn.Module class and works as a constructor for GRUs.

	   RNNModel class initiates a GRU module based on PyTorch's nn.Module class.
	   It has only two methods, namely init() and forward(). While the init()
	   method initiates the model with the given input parameters, the forward()
	   method defines how the forward propagation needs to be calculated.
	   Since PyTorch automatically defines back propagation, there is no need
	   to define back propagation method.

	   Attributes:
		   hid_dim (int): The number of nodes in each layer
		   n_layers (str): The number of layers in the network
		   gru (nn.GRU): The GRU model constructed with the input parameters.
		   fc (nn.Linear): The fully connected layer to convert the final state
						   of GRUs to our desired output shape.

	"""
	def __init__(self, rnnParams):
	#def __init__(self, input_dim, hid_dim, n_layers, output_dim, dropout, rnn_type='GRU'):
		"""The __init__ method that initiates a GRU instance.

		Args (in dict: rnnParams)
			input_dim (int): The number of nodes in the input layer
			hid_dim (int): The number of nodes in each layer
			n_layers (int): The number of layers in the network
			output_dim (int): The number of nodes in the output layer
			dropout_rnn (float): The probability of nodes being dropped out (rnn)
			rnn_type (string): 'GRU' or 'LSTM'
			
			inp_dim_fc (int): Number of fully connected input block's inputs
			fc_in_sizes (list of int): The fully connected block layer sizes 
										(the last layer length must be equal to 'hid_dim_rnn'.
			dropout_fc (float): The dropout probability of the fully connected input block

		"""
		super(FC_RNN_FC_Model, self).__init__()

		# Fully connected input block parameters:
		self.inp_dim_fc = rnnParams['inp_dim_fc']
		self.fc_in_sizes = rnnParams['fc_in_sizes']
		self.dropout_fc = rnnParams['dropout_fc']

		# RNN block parameters:
		self.input_dim = rnnParams['input_dim_rnn']
		self.hid_dim = rnnParams['hid_dim_rnn']
		self.n_layers = rnnParams['n_layers_rnn']
		self.output_dim = rnnParams['output_dim']
		self.dropout_rnn = rnnParams['dropout_rnn']
		self.rnn_type = rnnParams['rnn_type']
		
		# Fully connected output block parameters: fc_out_inp_dim = 25 * (64 + 120)
		self.batch_size = rnnParams['batch_size']
		self.seq_len = rnnParams['seq_len']
		self.fc_out_inp_dim = self.batch_size * self.seq_len
		
		# CONT HERE ...
		
		
		
		
		#self.fc_out_inp_dim = 25 * (self.hid_dim + self.input_dim)
		print("fc_out_inp_dim = ", self.fc_out_inp_dim)
		self.fc_out_sizes = rnnParams['fc_out_sizes']
		
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout=self.dropout_rnn, batch_first=True)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(self.input_dim, self.hid_dim, self.n_layers, dropout=self.dropout_rnn, batch_first=True)
			
		self.fc_seq_in = FC_Sequential(self.inp_dim_fc, self.fc_in_sizes, self.dropout_fc)
		#self.fc_Sequential = FC_Sequential(self.inp_dim_fc, self.fc_in_sizes, self.dropout_fc)

		# GRU layers
		#self.gru = nn.GRU(input_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)

		# Fully connected output layer:
		self.fc_Seq_out = FC_Sequential(self.fc_out_inp_dim, self.fc_out_sizes, self.dropout_fc)
		#self.fc_out = nn.Linear(self.hid_dim, self.output_dim)

	def forward(self, inp_seq, inp_fc):
		#print("RNNModel: inp_seq.shape = ", inp_seq.shape)
		#print("RNNModel: inp_fc.shape = ", inp_fc.shape)
		
		"""The forward method takes input tensor 'inp_seq' and does forward propagation

		Args:
			inp_seq (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

			inp_fc (torch.Tensor): The inputs to the fully connected block; shape (batch size, inp_dim_fc)

		Returns:
			outputs (torch.Tensor): The output tensor of the shape (batch_size, seq_len, output_dim)

		"""
		# Initializing hidden state for first input the output from the fully
		# connected network (inputting static forest variable data):
		# H0 dimensions shall be: [n_layers,batch_size,hid_dim)]
		
		h0 = torch.zeros(self.n_layers, inp_seq.size(0), self.hid_dim, device=inp_seq.device)
		if self.rnn_type == 'LSTM':
			c0 = torch.zeros(self.n_layers, inp_seq.size(0), self.hid_dim, device=inp_seq.device)
		
		for ii in range(self.n_layers):
			h0[ii,:,:] = self.fc_seq_in(inp_fc)
			#h0[ii,:,:] = self.fc_seq_in(inp_fc).unsqueeze(1)
			
			# If RNN = LSTM, initialize cell state with FC block outputs also:
			if self.rnn_type == 'LSTM':
				c0[ii,:,:] = self.fc_seq_in(inp_fc)
				#c0[ii,:,:] = self.fc_seq_in(inp_fc).unsqueeze(1)
		
		# Initializing hidden state for first input with zeros
		#h0 = torch.zeros(self.n_layers, inp_seq.size(0), self.hid_dim,device=inp_seq.device).requires_grad_()
		
		# Forward propagation by passing in the input and hidden state into the model
		# The outputs should be in the shape: [batch_size, seq_length, hid_size]
		if self.rnn_type == 'GRU':
			outputs, _ = self.rnn(inp_seq, h0)
		else:
			outputs, (_,_) = self.rnn(inp_seq, (h0, c0))
		
		#if self.rnn_type == 'GRU':
		#	outputs, _ = self.rnn(inp_seq, h0.detach())
		#else:
		#	outputs, (_,_) = self.rnn(inp_seq, (h0.detach(), c0.detach()))
			
		print("RNNModel: outputs.shape = ", outputs.shape)
		
		#outputs, _ = self.gru(inp_seq, h0.detach())

		# Reshaping the outputs in the shape of (batch_size, seq_length, hid_size)
		# so that it can fit into the fully connected layer
		#outputs = outputs[:, -1, :]
		
		# Eiku, ei noita kahta seuraavaa riviä kuitenkaan (output muuttujat
		# dimensiossa 2):
		#outputs = torch.flatten(outputs, start_dim=1, end_dim=-1)
		#inp_seq_flat = torch.flatten(inp_seq, start_dim=1, end_dim=-1)
		
		# --------------------------------------------------------------
		# Idea: concatenate input sequence and the RNN outputs to the
		# input of the output FC block (4.2.2024)
		# 
		# Concatenate the RNN outputs and input sequence (in dim = 2)
		# to be fed to the output fully connected block. For instance,
		# if the hidden dimension is e.g. 64 and the encoder input dimension
		# input_dim_enc = 120, the input dimension of the output FC block
		# should be 25 * (64 + 120) = 4600 (is this correct?)
		fc_out_in = torch.cat((outputs, inp_seq), dim=2)
		print("fc_out_in.shape = ", fc_out_in.shape)
		# --------------------------------------------------------------

		# Convert the final state to our desired output shape (batch_size, seq_len, output_dim)
		outputs = self.fc_Seq_out(fc_out_in)
		#outputs = self.fc_out(outputs)

		return outputs


# The encoder code imodified fron original code in:
#
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
# 
# 
# MIT License
#
# Copyright (c) 2018 Ben Trevett
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.    
    
class Encoder(nn.Module):
	def __init__(self, input_dim, hid_dim, n_layers, dropout, rnn_type='GRU'):
		super(Encoder, self).__init__()
		
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		
		self.rnn_type = rnn_type
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
			
		self.dropout = nn.Dropout(dropout)

		
	def forward(self, src):
	
		#print("Encoder: src.shape = ", src.shape)
		
		# With batch_first=True the dimensions are:
		# encInput: [batch_size, seq_len, input_dim]
		encInput = self.dropout(src)

		# Initializing hidden state (h0, c0) for first input with zeros
		h0 = torch.zeros(self.n_layers, encInput.size(0), self.hid_dim, device=src.device)
		if self.rnn_type == 'LSTM':
			c0 = torch.zeros(self.n_layers, encInput.size(0), self.hid_dim, device=src.device)

		if self.rnn_type == 'GRU':
			outputs, hidden = self.rnn(encInput, h0)
			cell = None
		else:
			outputs, (hidden, cell) = self.rnn(encInput, (h0, c0))
		
		# With batch_first=True and n directions is always one:
		# outputs = [batch size, seq_len, hid dim]
		# hidden = [n layers, batch size, hid dim]
		# cell = [n layers, batch size, hid dim]
		
		# outputs are always from the top hidden layer
		# The context vector will now be both the final hidden state and the final cell state
		# returning only hidden & cell states here:
		
		return hidden, cell



# The fully connected block structure is defined by the input parameters
# 'hid_sizes'. The first element (hid_sizes[0]) gives the size of the first
# hidden layer (not the input layer for which the size is 'inp_dim_fc'), and
# the last element the size of the output layer (hid_sizes[nr_hidden-1]).

class FC_Sequential(nn.Module):
	def __init__(self, inp_dim_fc, hid_sizes, dropout):
	#def __init__(self, inp_dim_fc, nr_hidden, hid_sizes, dropout):
		super(FC_Sequential, self).__init__()
		
		layers = []
		for i in range(len(hid_sizes)):
		#for i in range(nr_hidden):
			layers.append(nn.Linear(inp_dim_fc if i == 0 else hid_sizes[i-1], hid_sizes[i]))
			#layers.append(nn.BatchNorm1d(hid_sizes[i]))
			layers.append(nn.ReLU(inplace = True))
			#layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout))
			
		self.fc_layers = nn.Sequential(*layers)
	   
	def forward(self, inp_fc):
		# output = [batch_size, hid_sizes[-1]]
		output = self.fc_layers(inp_fc)
		return output


class Combined(nn.Module):
	def __init__(self, enc_params, fc_params):
	#def __init__(self, input_dim, hid_dim, n_layers, dropout, nr_hidden, hid_sizes, rnn_type='GRU'):
		super(Combined, self).__init__()
		
		# input_dim, hid_dim, n_layers, dropout, , rnn_type='GRU'
		self.input_dim_enc = enc_params['input_dim_enc']
		self.hid_dim_enc = enc_params['hid_dim_enc']
		self.n_layers_enc = enc_params['n_layers_enc']
		self.dropout_enc = enc_params['dropout_enc']
		self.rnn_type = enc_params['rnn_type']
		
		# Fully connected block parameters:
		#  fc_in_sizes, dropout_fc
		self.inp_dim_fc = fc_params['inp_dim_fc']
		#self.nr_hid_fc = fc_params['nr_hid_fc']
		self.fc_in_sizes = fc_params['fc_in_sizes']
		self.dropout_fc = fc_params['dropout_fc']
		
		# Define the combined module hidden (and cell) dimension:
		if self.rnn_type == 'GRU':
			self.hid_dim_combi = self.hid_dim_enc + self.fc_in_sizes[-1]/self.n_layers_enc
			#self.hid_dim_combi = self.hid_dim_enc + self.fc_in_sizes[-1]
		else:
			self.hid_dim_combi = self.hid_dim_enc + self.fc_in_sizes[-1]/self.n_layers_enc
			#self.hid_dim_combi = self.hid_dim_enc + self.fc_in_sizes[-1]/2
		
		self.encoder = Encoder(self.input_dim_enc, self.hid_dim_enc, self.n_layers_enc, self.dropout_enc, self.rnn_type)
		
		self.fc_Sequential = FC_Sequential(self.inp_dim_fc, self.fc_in_sizes, self.dropout_fc)
		
	def forward(self, src, static_data, target=None):
	
		if self.rnn_type == 'GRU':
			#print("Combined: src.shape = ", src.shape)
			hidden, cell = self.encoder(src)
		else:
			hidden, cell = self.encoder(src)
		
		# Is the squeeze wrong? How about batch dimension? 
		fc_output = self.fc_Sequential(static_data).squeeze()
		
		# Concatenate the output of the fully connected block with the
		# hidden abd cell states from the RNN encoder block to generate
		# a combined context vectors to be passed to the decoder module
		
		# The output dimensions of the fully connected block have to be
		# adjusted to the output shape of the RNN modules. The output 
		# shape of the fully connected moduel is:
		#
		# fc_output = [batch_size, fc_in_sizes[-1]]
		#
		# and for the hidden & cell state of the GRU % LSTM modules 
		# (from encoder):
		#
		# hidden = [n layers * n_directions, batch size, hid dim]
		# cell = [n layers * n_directions, batch size, hid dim]
		#
		# NOTE: n directions = 1 (always), so the sizes are:
		#
		# hidden = [n layers, batch size, hid dim]
		# cell = [n layers, batch size, hid dim]
		#
		# Consequently the outputs from the fully connected block
		# (nbr of outputs = fc_in_sizes[-1]) has to be split into
		# as many layers as there are in GRU or LSTM. 
		
		# Ensure that batch dimension is there.
		# Check that the number of dimensions in fc_output is 2
		# (with test set batch_size is 1, so the nr dim = ):
		if fc_output.dim() < 2:
			fc_output = torch.unsqueeze(fc_output, 0)
		
		fc_output = reshapeFcOutput(fc_output, self.n_layers_enc)
		
		# Obsolete: (Concatenate half of the fully connected outputs
		# to LSTM hidden, and the other half to the LSTM cell state; with
		# GRU concatenate all the fully connected outputs to the hidden state.
		# Note that the hidden dimension of the decoder has to be defined
		# as hid_dim_dec = hid_dim_enc + fc_in_sizes[-1]/2, and thus the
		# output layer size in the fully connected block must be a multiple
		# of two.)
		
		# The concatenation shall be done in dim = 2 (= third dimension), 
		# as the hidden and cell tensor dimensions are: 
		# hidden = [n_layers * n_directions, batch size, hid dim]
		# As n_directions = 1, then:
		#
		# hidden = [n_layers, batch size, hid dim]
		#
		# With LSTM concatenate the fc_output to both hidden and cell
		# state inputs of the decoder:
		if self.rnn_type == 'GRU':
			#print("Combined: hidden.shape (pre cat) = ", hidden.shape)
			#print("Combined: fc_output.shape (pre cat) = ", fc_output.shape)
			hidden = torch.cat((hidden, fc_output), dim=2)
			#print("Combined: hidden.shape (post cat) = ", hidden.shape)
			cell = None
		else:
			hidden = torch.cat((hidden, fc_output), dim=2)
			# The same
			cell = torch.cat((cell, fc_output), dim=2)
			# hidden = torch.cat((hidden, fc_output[0:self.fc_in_sizes[-1]/2]), dim=2)
			# cell = torch.cat((cell, fc_output[self.fc_in_sizes[-1]/2:]), dim=2)

		return hidden, cell
		

class Decoder(nn.Module):
	def __init__(self, output_dim, inp_dim, hid_dim, n_layers, dropout, rnn_type='GRU'):
		super(Decoder, self).__init__()
		
		self.output_dim = output_dim
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		
		self.rnn_type = rnn_type
		
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(inp_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(inp_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)
		
		self.fc_out = nn.Linear(hid_dim, output_dim)
		
		self.dropout = nn.Dropout(dropout)

		
	def forward(self, input_dec, hidden, cell):
		
		#print("Decoder: input_dec.shape 1 = ", input_dec.shape)
		
		# With batch_first=True and n_directions = 1 always: 
		# input_dec: [batch_size, input_dim]
		# hidden = [n_layers , batch_size, hid_dim]
		# cell = [n_layers, batch_size, hid_dim]
		
		# unsqueeze(...) Returns a new tensor with a dimension of 
		# size one inserted at the specified position (here: second 
		# dimension, as we defined batch_first = True, and the first 
		# dimension is for batch):
		#
		# input_dec = [batch_size, 1, input_dim]
		
		input_dec = input_dec.unsqueeze(1)
		
		#print("Decoder: input_dec.shape 2 = ", input_dec.shape)
		#print("Decoder: hidden.shape = ", hidden.shape)
		
		decInput = self.dropout(input_dec)
		
		if self.rnn_type == 'GRU':
			output, hidden = self.rnn(decInput, hidden)
			cell = None
		else:
			output, (hidden, cell) = self.rnn(decInput, (hidden, cell))
		
		#print("Decoder: output.shape 1 = ", output.shape)
		#print("Decoder: hidden.shape = ", hidden.shape)
		
		# Note: The output tensor dimensions of torch.nn.GRU or torch.nn.LSTM are:
		#
		# (N, L, D x Hout), (batch_first = True) where 
		#
		# N = batch size
		# L = sequence length
		# D = 2 if bidirectional = True otherwise 1
		# Hout = hidden_size
		#
		# see: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
		# or: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
		# ---------------------------------------------------------------------
		
		# seq_len and n_directions will always be 1 in the decoder, therefore:
		# output = [batch_size, 1, hid_dim]
		# hidden = [n_layers, batch_size, hid_dim]
		# cell = [n_layers, batch_size, hid_dim]
		
		# ---------------------------------------------------------------------
		# Here the squeeze operation removes the first dimension of the rnn
		# output, which is [seq_len, batch-size, hidden_dim], 
		# the resulting tensor has then the dimensions [batch-size, hidden_dim]:
		
		#output = output.squeeze(0)
		prediction = self.fc_out(output)
		#prediction = self.fc_out(output.squeeze(0))
		
		#print("Decoder: output.shape 2 = ", output.shape)
		#print("Decoder: prediction.shape = ", prediction.shape)
		
		# Note: seq_len and n_directions will always be 1 in the decoder!
		# The dimensions of 'prediction' thus are:
		#
		# prediction = [batch_size, 1, output_dim]
		#
		# If only one variable will be predicted with the model, then
		# output_dim = 1.
		
		# Remove the second dimension (seq_len), as it is always 1:
		return prediction.squeeze(1), hidden, cell
		#return prediction, hidden, cell
		


# The Seq2Seq code imodified fron original code in:
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
#
# MIT License
#
# Copyright (c) 2018 Ben Trevett
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# The concept is to
#
# 1. Provide the inputs - the daily weather data time series, and the static_data
# site dependent data to the combined encoder/fully connected block
# 2. Then to produce the yearly prediction for the selected nbr of years in 
# a loop (as with the translation example). The target length is the sequence
# length, i.e. the number of years to predict (default = 25).
#
# As the task is regression type, the output of the decoder is a single real
# number (per forest/carbon variable), and thus there is no need for the 
# argmax() operation to obtain the output.

class Seq2Seq(nn.Module):
	def __init__(self, combined, decoder, device):
		super().__init__()
		
		self.combined = combined
		self.decoder = decoder
		self.device = device
		
		#print("combined.hid_dim_combi = ", combined.hid_dim_combi)
		#print("decoder.hid_dim = ", decoder.hid_dim)
		
		assert combined.hid_dim_combi == decoder.hid_dim, \
			"Hidden dimensions of encoder and decoder must be equal!"
		assert combined.n_layers_enc == decoder.n_layers, \
			"Encoder and decoder must have equal number of layers!"
		
	def forward(self, src, static_data, trg, teacher_forcing_ratio = 0.5):
	
		# src: [batch_size, seq_len, output_dim]
		# trg = [batch_size, seq_len]
		#teacher_forcing_ratio is probability to use teacher forcing
		#e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
		
		batch_size = trg.shape[0]
		trg_len = trg.shape[1]
		trg_dim = self.decoder.output_dim
		#trg_vocab_size = self.decoder.output_dim
		
		#tensor to store decoder outputs
		outputs = torch.zeros(batch_size, trg_len, trg_dim).to(self.device)
		#outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
		
		#last hidden state of the combined module is used as the initial hidden state of the decoder
		hidden, cell = self.combined(src, static_data)
		
		# First input to the decoder (if teacher_force, then
		# always trg[:,0,:])
		# Decide if we are going to use teacher forcing or not
		
		#input_dec = trg[:,0,:]
		teacher_force = random.random() < teacher_forcing_ratio
		input_dec = trg[:,0,:] if teacher_force else torch.zeros(trg[:,0,:].shape).to(self.device)
		
		#print("Seq2Seq: trg.shape = ", trg.shape)
		#print("Seq2Seq: input_dec.shape = ", input_dec.shape)
		#print("Seq2Seq: hidden.shape = ", hidden.shape)

		for t in range(0, trg_len):
			
			#insert input token embedding, previous hidden and previous cell states
			#receive output tensor (predictions) and new hidden and cell states
			
			prediction, hidden, cell = self.decoder(input_dec, hidden, cell)
			#output, hidden, cell = self.decoder(input_dec, hidden, cell)
			
			#print("Seq2Seq; for-loop: prediction.shape = ", prediction.shape)
			#print("Seq2Seq; for-loop: hidden.shape = ", hidden.shape)
			
			# Place predictions in a tensor holding predictions for each year:
			outputs[:,t,:] = prediction
			#outputs[:,t,:] = prediction.squeeze(1)
			
			#decide if we are going to use teacher forcing or not
			teacher_force = random.random() < teacher_forcing_ratio
			
			#if teacher forcing, use actual next token as next input
			#if not, use predicted token
			# ---------------------------------------------------------------------
			# Note: If more than one target variable is to be predicted, then
			# should the target dimension be: [trg_len, batch_size, trg_dim]
			# and the next row: input_dec = trg[:,t,:] if teacher_force else output
			# ---------------------------------------------------------------------
			input_dec = trg[:,t,:] if teacher_force else prediction
			#input_dec = trg[:,t,:] if teacher_force else prediction.squeeze(1)
		
		return outputs


# train()
#
# Seq2seq model training function

def train(model, iterator, optimizer, criterion, clip):
    
	model.train()
	
    # Use data containers for RMSE computation:
	targetsTrain = numpy.empty((len(dataset_train),nrOutputs), dtype=np.float32)
	predsTrain = numpy.empty((len(dataset_train),nrOutputs), dtype=np.float32)

	epoch_loss = 0

	for i, dataItem in enumerate(iterator):
		inputs_enc = dataItem['inputDataEnc']
		inputs_fc = dataItem['inputDataFc']
		trg = dataItem['targetData']
				
		optimizer.zero_grad()

		output = model(inputs_enc, inputs_fc, trg)

		#trg = [trg len, batch size]
		#output = [trg len, batch size, output dim]

		# The ouput dimension is the dimension along the last axis:
		output_dim = output.shape[-1]
		
		output = output.view(-1, output_dim)
		trg = trg.view(-1, output_dim)
		#trg = trg.view(-1)

		# The original code dropped the first items in output & trg:
		#output = output[1:].view(-1, output_dim)
		#trg = trg[1:].view(-1)
	   
		#trg = [trg len * batch size]
		#output = [trg len * batch size, output dim]
		##trg = [(trg len - 1) * batch size]
		##output = [(trg len - 1) * batch size, output dim]

		#print("train: output.shape = ", output.shape)
		#print("train: trg.shape = ", trg.shape)
		
		loss = criterion(output, trg)
		
		loss.backward()
		
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		
		optimizer.step()
		
		epoch_loss += loss.item()
		
		print('train ' + str(i) + ': ' + str(loss), end="")
		print("\r", end="")

	return epoch_loss / len(iterator)





# split_into_chunks()
#
# This function  splits a two-dimensional input tensor into N equally sized 
# chunks along the first axis (axis = 0). This function is used when organizing
# the fully connected block's outputs to be concatenated with the RNN encoder  
# module hidden (& cell with LSTM) state.

def split_into_chunks(tensor, num_chunks):
    # Check if the number of chunks is valid
    assert tensor.size(0) % num_chunks == 0, "Number of chunks must evenly divide the size along axis 0"

    # Calculate the size of each chunk
    chunk_size = tensor.size(0) // num_chunks

    # Use the chunk function to split the tensor into chunks along axis 0
    chunks = torch.chunk(tensor, chunks=num_chunks, dim=0)

    return chunks


# reshapeFcOutput()
#
# This function reshapes the fully connected block's output to
# match the size of the RNN module hidden layer. See Combined()
# for usage.

def reshapeFcOutput(inputTensor, n_layers_enc):

	# The input tensor dimensions are:
	# dim_fc = [batch_size, fc_in_sizes[-1]]
	
	# Transpose the input tensor to dim = [fc_in_sizes[-1], batch_size]
	#print(inputTensor.shape)
	inputTensor = inputTensor.transpose(0,1)

	# Specify the number of chunks (along first dimension)
	#num_chunks = 4

	# Split the tensor into N equally sized chunks
	chunks = split_into_chunks(inputTensor, n_layers_enc)

	# Print the resulting chunks
	#for i, chunk in enumerate(chunks):
	#	print(f"Chunk {i + 1}:\n{chunk}\n")

	# Transpose the chunks back to dim = [batch_size, fc_in_sizes[-1] / n_layers_enc],
	# add one dimension (first), and concatenate (along first axis) to obtain a 
	# three-dimensional output tensor:
	for i, chunk in enumerate(chunks):
		chunk = chunk.transpose(0,1)
		chunk = chunk.unsqueeze(0)
		if i==0:
			outPut = chunk
		else:
			outPut = torch.cat((outPut, chunk), dim=0)

	# for ii in range(outPut.shape[0]):
		# print(outPut[ii,:,:])

	# The output dimensions are:
	# outPut = [n_layers, batch size, hid dim]
	# which equals the decoder hidden input dimensios. 

	return outPut


# ========================================================================
# Transformer model
#
# Original code from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#
# New link (the above leads to different page):
# https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
# ========================================================================
#
# model = TransformerModel(output_dim, d_model, nhead, d_hid, nlayers, dropout).to(device)
#
# self.output_dim = tfParams['output_dim_tf']
# self.d_model = tfParams['d_model_tf']
# self.nhead = tfParams['nhead_tf']
# self.hid_dim = tfParams['hid_dim_tf']
# self.dropout = tfParams['dropout_tf']
# self.nlayers = tfParams['nlayers_tf']

class TransformerModel(nn.Module):

	## def __init__(self, output_dim: int, d_model: int, nhead: int, hid_dim: int,
	##			 nlayers: int, dropout: float = 0.5):
	def __init__(self, tfParams):
		super().__init__()
		
		# Input dimension:
		self.output_dim = tfParams['output_dim_tf']
		self.d_model = tfParams['d_model_tf']
		self.nhead = tfParams['nhead_tf']
		self.hid_dim = tfParams['hid_dim_tf']
		self.dropout = tfParams['dropout_tf']
		self.nlayers = tfParams['nlayers_tf']
		
		self.inp_dim_fc = tfParams['inp_dim_fc']
		
		# The nYears dictates the length of the seq_len dimension.
		# As the static input variables (forest vars + site info) are concatenated
		# with the sequential climate data inputs, the 'seq_len' dimension has to be
		# increased with one or two (depending on the number of features in static 
		# and climate data):
		nYears = tfParams['nYears']
		if self.inp_dim_fc <= self.d_model:
			self.seq_len = nYears + 1
		else:
			self.seq_len = nYears + 2
			
		self.model_type = 'Transformer'
        
		self.pos_encoder = PositionalEncoding(self.d_model, self.dropout, self.seq_len)
		#self.pos_encoder = PositionalEncoding_mod(self.d_model, self.dropout, self.seq_len)
        
		encoder_layers = TransformerEncoderLayer(self.d_model, self.nhead, self.hid_dim, self.dropout, batch_first = True)
		self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
		#self.embedding = nn.Embedding(self.output_dim, self.d_model)
		print("self.output_dim = ", self.output_dim)
		print("self.d_model = ", self.d_model)
		print("self.nhead = ", self.nhead)
		print("self.hid_dim = ", self.hid_dim)
		print("self.dropout = ", self.dropout)
		print("self.nlayers = ", self.nlayers)
		self.linear = nn.Linear(self.d_model, self.output_dim)

		self.init_weights()

	def init_weights(self) -> None:
		initrange = 0.1
		#self.embedding.weight.data.uniform_(-initrange, initrange)
		self.linear.bias.data.zero_()
		self.linear.weight.data.uniform_(-initrange, initrange)

	def forward(self, src: Tensor, src_fc: Tensor, src_mask: Tensor = None) -> Tensor:
		"""
		Arguments:
			src: Tensor, shape [batch_size, seq_len, d_model] - batch first = True
			src_fc: tensor, shape [batch_size, inp_dim_fc]
			src_mask: Tensor, shape [batch_size, seq_len, d_model] - batch first = True
			## src: Tensor, shape ``[seq_len, batch_size]``
			## src_mask: Tensor, shape ``[seq_len, seq_len]``

		Returns:
			output Tensor of shape ``[batch_size, seq_len, output_dim]``
			## output Tensor of shape ``[seq_len, batch_size, output_dim]``
		"""
		#src = self.embedding(src) * math.sqrt(self.d_model)

		# ---------------------------------------------------------------------------
		# Add here the concatenation of the static inputs (forest vars + 
		# site info) with the climate data (sequential) input:
		
		# src_fc dimension: [batch_size, inp_dim_fc]
		#
		# Concatenate the static inputs (src_fc) with the sequential ones (src)
		# 
		# Note that with climate data yearly averages the input dimension d_model = 12,
		# but the length of static inputs may be longer (e.g. inp_dim_fc = 23). In this 
		# case the static inputs must be split into two layers (in seq_len dimension)
		# of the produced scr.

		if self.inp_dim_fc <= self.d_model:
			# Pad the static inputs with zeros to match the input dimension (d_model),
			# and add seq_len dimension:
			padd = nn.ZeroPad2d((0,self.d_model-self.inp_dim_fc,0,0))
			static_inputs = padd(src_fc).unsqueeze(1)

			# In this case insert the static inputs as the first layer of src:
			#static_inputs = torch.zeros(src.shape[0], 1, self.d_model, device=src.device)
			#static_inputs[:,0,0:self.inp_dim_fc] = src_fc.unsqueeze(1)
		else:
			# Else split src_fc into two layers:
			splitIdx = int(np.ceil(self.inp_dim_fc/2))

			padd = nn.ZeroPad2d((0,self.d_model-splitIdx,0,0))
			s1 = src_fc[:,0:splitIdx]
			static_inputs = padd(s1).unsqueeze(1)
			#print("s1.shape = ", s1.shape)
			#print("static_inputs.shape = ", static_inputs.shape)
			
			padd = nn.ZeroPad2d((0,self.d_model-splitIdx+1,0,0))
			s2 = src_fc[:,splitIdx::]
			static_inputs = torch.cat((static_inputs, padd(s2).unsqueeze(1)), dim=1)
			#print("s2.shape = ", s2.shape)
			#print("static_inputs.shape = ", static_inputs.shape)

			'''
			static_inputs = torch.zeros(src.shape[0], 2, self.d_model, device=src.device)
			print("static_inputs.shape = ", static_inputs.shape)
			splitIdx = int(np.ceil(self.inp_dim_fc/2))
			print("splitIdx = ", splitIdx)
			s1 = src_fc[:,0:splitIdx].unsqueeze(1)
			s2 = src_fc[:,splitIdx:self.inp_dim_fc].unsqueeze(1)
			print("s1.shape = ", s1.shape)
			print("s2.shape = ", s2.shape)
			
			print("static_inputs[:,0,0:splitIdx].shape = ", static_inputs[:,0,0:splitIdx].shape)
			print("static_inputs[:,1,splitIdx:self.inp_dim_fc].shape = ", static_inputs[:,1,splitIdx:self.inp_dim_fc].shape)
			static_inputs[:,0,0:splitIdx] = s1
			static_inputs[:,1,splitIdx:self.inp_dim_fc] = s2
			'''
		src = torch.cat((static_inputs, src), dim = 1)
	   
		# ---------------------------------------------------------------------------
		
		src = self.pos_encoder(src)
		
		if src_mask is None:
			pass
			"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
			Unmasked positions are filled with float(0.0).
			"""
			#src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1])
			##src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
		
		output = self.transformer_encoder(src)
		#output = self.transformer_encoder(src, src_mask)
		output = self.linear(output)
		
		return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 27):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        
        # Test different decays with div_term:
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(5.0) / d_model))
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model))
        # #div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        #print("position.shape ", position.shape)
        #print("div_term.shape ", div_term.shape)
        #print("pe.shape ", pe.shape)
        #print("pe[0,:,:] = ", pe[0,:,:])
        
        pe = pe.to(self.device)
        
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, input_dim]``
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class PositionalEncoding_mod(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 27):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        
        # Note the nbr of climate vars is hard coded here! (to be modified!):
        # If d_model == nrClimVars, then the yearly climate variables are given
        # as inputs:
        nrClimVars = 8
        if d_model == nrClimVars:
            div_term = torch.linspace(1, 0.1, 4)
        else:
            # Else the monthly climate variables are given ():
            aa = torch.linspace(1, 0.1, 6).unsqueeze(0)
            div_term = aa
            for ii in range(nrClimVars-1):
                div_term = torch.cat((div_term, aa), 1)
                
            div_term = div_term.squeeze(0)
        
        # PositionalEncoding() v.1.1
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model))
        # PositionalEncoding() v.1.0 = original code:
        # #div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        
        # Use only the position encoding along the sequence dimension (no div_term here):
        # PositionalEncoding() v.1.2
        # pe[0, :, 0::2] = torch.sin(position)
        # pe[0, :, 1::2] = torch.cos(position)
        
        # Original code:
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        #print("position.shape ", position.shape)
        #print("div_term.shape ", div_term.shape)
        #print("pe.shape ", pe.shape)
        #print("pe[0,:,:] = ", pe[0,:,:])
        
        pe = pe.to(self.device)
        
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, input_dim]``
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


# initModel()
#
# This function defines the model structure, initializes the model weights and
# specifys the loss function and optimizer.
#
# Inputs:
#
# paramDict		(dict) The dictionary holding the input parameters read from
#				parameter file.
#
# verbose		(boolean) If True, then some information will be printed.


def initModel(paramDict, verbose = True):

	# The encoder input dimension is the <nbr of climate variables = K> x <nbr aggregate items>
	# e.g. input_dim_enc = K x 24 = 120 (bi-monthly), K x 12 (monthly), or K x 1 (yearly)
	#
	# Presently, a common selection of input climate variables is:
	# climDataCols = PAR_sum TAir_sum Precip_sum VPD_sum PAR_mean TAir_mean Precip_mean VPD_mean CO2_mean TAir_std Precip_std VPD_std
	#
	# So the number of climate variables K = 12

	# If the model to be initialized is a cascade model, then the cascade input
	# variables (time series variables of lrngth = nYears) will be concatenated
	# with the climate variables, and the input dimension of the encoder have to
	# be increased by the number of these variables (i.e. affects input_dim_enc,
	# or d_model_tf):
	if paramDict['cascadeInputVars'] is not None:
		nrCascadevars = int(len(paramDict['cascadeInputVars'])/paramDict['nYears'])
	else:
		nrCascadevars = 0
	print("nrCascadevars = ", nrCascadevars)

	# Use the encoder parameters for the FC_RNN_Model also:
	input_dim_enc = paramDict['input_dim_enc'] + nrCascadevars
	hid_dim_enc = paramDict['hid_dim_enc']
	n_layers_enc = paramDict['n_layers_enc']
	dropout_enc = paramDict['dropout_enc']
	rnn_type = paramDict['rnn_type']

	# The input dimension of the fully connected block:
	# siteInfo parameters: 10 (or 8 without nLayers & nSpecies)
	# forest variables: 12 (species-wise variables: age, H. D, BA
	inp_dim_fc = len(paramDict['inputVarFcCols'])

	# Note (seq2seq): The number of neurons in the fully connected block's
	# outputlayer must be integer divisable with the number of layers in the
	# decoder (& encoder), as the FC block's outputs will be split into
	# decoder's different layers' hidden inputs (& cell inputs with LSTM):
	fc_in_sizes = paramDict['fc_in_sizes']
	dropout_fc = paramDict['dropout_fc']

	# Decoder output dimension (number of predicted variables).
	# Note: This input applies alos to FC_RNN_Model.
	output_dim = len(paramDict['targetVars'])

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if paramDict['modelType'] == 'S2S':
		
		enc_params = {
			'input_dim_enc': input_dim_enc,
			'hid_dim_enc': hid_dim_enc,
			'n_layers_enc': n_layers_enc,
			'dropout_enc': dropout_enc,
			'rnn_type': rnn_type
			}
		
		fc_params = {
			'inp_dim_fc': inp_dim_fc,
			'fc_in_sizes': fc_in_sizes,
			'dropout_fc': dropout_fc
			}
		
		# The input dimension is the same as the output dimension (nbr of target variables)
		inp_dim_dec = output_dim

		# The encoder hidden state and the fully connected block's output layer
		# will be concatenated and fed into the first hidden state (h0, c0) of 
		# the decoder. If the decoder (& encoder) includes several layers, the
		# FC block outputs will be split to as may equal-sized parts, and distributed
		# evenly to the separate decoder layers:
		hid_dim_dec = hid_dim_enc + int(fc_in_sizes[-1]/n_layers_enc)
		
		n_layers_dec = n_layers_enc
		dropout_dec = paramDict['dropout_dec']
		
		if verbose:
			print("hid_dim_enc = ", hid_dim_enc)
			print("hid_dim_dec = ", hid_dim_dec)
		
		comb = Combined(enc_params, fc_params)
		dec = Decoder(output_dim, inp_dim_dec, hid_dim_dec, n_layers_dec, dropout_dec, rnn_type=rnn_type)

		# Initialize the model:
		model = Seq2Seq(comb, dec, device).to(device)
		
	elif paramDict['modelType'] == 'FC_RNN':

		n_layers_fc2h0 = paramDict['n_layers_fc2h0']
		
		# Next line obsolete with this type of model:
		#fc_out_sizes = paramDict['fc_out_sizes']
		
		if fc_in_sizes[-1] != hid_dim_enc:
			fc_in_sizes[-1] = hid_dim_enc
			if verbose:
				print("Forced FC input block last layer size to equal RNN module hidden dimension", hid_dim_enc)

		#if fc_out_sizes[-1] != len(paramDict['targetVars']):
		#    fc_out_sizes[-1] = len(paramDict['targetVars'])
		#    print("Forced FC output block last layer size to equal the number of target variables", hid_dim_enc)
		
		rnnParams = {
			'input_dim_rnn': input_dim_enc,
			'hid_dim_rnn': hid_dim_enc,
			'n_layers_rnn': n_layers_enc,
			'n_layers_fc2h0': n_layers_fc2h0,
			'dropout_rnn': dropout_enc,
			'output_dim': output_dim,
			'rnn_type': rnn_type,
			'inp_dim_fc': inp_dim_fc,
			'fc_in_sizes': fc_in_sizes,
			'dropout_fc': dropout_fc
			}
	   
		# Initialize the model:
		model = FC_RNN_Model(rnnParams).to(device)

	elif paramDict['modelType'] == 'XFORMER':
		# The number of expected features in the encoder/decoder inputs:

		# Source, target & output dimensions:
		# src: (N, S, E) if batch_first=True.
		# tgt: (N, T, E) if batch_first=True.
		#
		# output: (N, T, E) if batch_first=True.
		#
		# where S is the source sequence length, T is the target sequence length, 
		# N is the batch size, E is the feature number
		#
		# Note: The variable 'd_model_tf' (or 'd_model') corresponds to the 
		# variable 'input_dim_enc' of FC_RNN or seq2seq models. 
		d_model_tf = paramDict['d_model_tf'] + nrCascadevars

		tfParams = {
			'output_dim_tf': output_dim, 
			'd_model_tf': d_model_tf,
			'nhead_tf': paramDict['nhead_tf'],
			'hid_dim_tf': paramDict['hid_dim_tf'],
			'nlayers_tf': paramDict['nlayers_tf'],
			'nYears': paramDict['nYears'],
			'inp_dim_fc': inp_dim_fc,
			'dropout_tf': paramDict['dropout_tf']
			}

		model = TransformerModel(tfParams).to(device)
		
	elif paramDict['modelType'] == 'FC_RNN_FC':

		n_layers_fc2h0 = paramDict['n_layers_fc2h0']
		
		# Next line obsolete with this type of model:
		#fc_out_sizes = paramDict['fc_out_sizes']
		
		if fc_in_sizes[-1] != hid_dim_enc:
			fc_in_sizes[-1] = hid_dim_enc
			if verbose:
				print("Forced FC input block last layer size to equal RNN module hidden dimension", hid_dim_enc)

		#if fc_out_sizes[-1] != len(paramDict['targetVars']):
		#    fc_out_sizes[-1] = len(paramDict['targetVars'])
		#    print("Forced FC output block last layer size to equal the number of target variables", hid_dim_enc)
		
		rnnParams = {
			'input_dim_rnn': input_dim_enc,
			'hid_dim_rnn': hid_dim_enc,
			'n_layers_rnn': n_layers_enc,
			'n_layers_fc2h0': n_layers_fc2h0,
			'dropout_rnn': dropout_enc,
			'output_dim': output_dim,
			'rnn_type': rnn_type,
			'inp_dim_fc': inp_dim_fc,
			'fc_in_sizes': fc_in_sizes,
			'fc_out_sizes': paramDict['fc_out_sizes'],
			'dropout_fc': dropout_fc
			}
	   
		# Initialize the model:
		model = FC_RNN_FC_Model(rnnParams).to(device)

	model.apply(init_weights)

	if verbose:
		print(model)
		print(f'The model has {count_parameters(model):,} trainable parameters')

	learning_rate = paramDict['learning_rate']

	if paramDict['loss_function'] == 'CustomLoss':
		criterion = CustomLoss(rmseFactor = paramDict['rmseFactor'], biasFactor = paramDict['biasFactor'], r2limit = paramDict['r2limit'])
	elif paramDict['loss_function'] == 'CustomLoss_perCase':
		criterion = CustomLoss2(rmseFactor = paramDict['rmseFactor'], biasFactor = paramDict['biasFactor'], r2limit = paramDict['r2limit'])
	elif paramDict['loss_function'] == 'CustomLoss_perYear':
		criterion = CustomLoss_perYear(rmseFactor = paramDict['rmseFactor'], biasFactor = paramDict['biasFactor'], r2limit = paramDict['r2limit'])
	else:
		criterion = nn.MSELoss()

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
	#scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=30)

	return model, criterion, optimizer, scheduler









class TransformerModel_foo(nn.Module):

	## def __init__(self, output_dim: int, d_model: int, nhead: int, hid_dim: int,
	##			 nlayers: int, dropout: float = 0.5):
	def __init__(self, tfParams):
		super().__init__()
		
		# Input dimension:
		self.output_dim = tfParams['output_dim_tf']
		self.d_model = tfParams['d_model_tf']
		self.nhead = tfParams['nhead_tf']
		self.hid_dim = tfParams['hid_dim_tf']
		self.dropout = tfParams['dropout_tf']
		self.nlayers = tfParams['nlayers_tf']
		
		# The nYears dictates the length of the postional encoding seq_len dimension:
		self.nYears = tfParams['nYears']
		
		self.model_type = 'Transformer'
		self.pos_encoder = PositionalEncoding(self.d_model, self.dropout, self.nYears)
		encoder_layers = TransformerEncoderLayer(self.d_model, self.nhead, self.hid_dim, self.dropout, batch_first = True)
		self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
		#self.embedding = nn.Embedding(self.output_dim, self.d_model)
		print("self.output_dim = ", self.output_dim)
		self.linear = nn.Linear(self.d_model, self.output_dim)

		self.init_weights()

	def init_weights(self) -> None:
		initrange = 0.1
		#self.embedding.weight.data.uniform_(-initrange, initrange)
		self.linear.bias.data.zero_()
		self.linear.weight.data.uniform_(-initrange, initrange)

	def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
		"""
		Arguments:
			src: Tensor, shape [batch_size, seq_len, d_model] - batch first = True
			src_mask: Tensor, shape [batch_size, seq_len, d_model] - batch first = True
			## src: Tensor, shape ``[seq_len, batch_size]``
			## src_mask: Tensor, shape ``[seq_len, seq_len]``

		Returns:
			output Tensor of shape ``[batch_size, seq_len, output_dim]``
			## output Tensor of shape ``[seq_len, batch_size, output_dim]``
		"""
		#src = self.embedding(src) * math.sqrt(self.d_model)
		
		# Add heer the concatenation of the static inputs (forest vars + 
		# site info) with the climate data (sequential) input:
		
		# CONTINUE HERE ...
		
		
		src = self.pos_encoder(src)
		
		if src_mask is None:
			pass
			"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
			Unmasked positions are filled with float(0.0).
			"""
			#src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1])
			##src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
			
		output = self.transformer_encoder(src)
		#output = self.transformer_encoder(src, src_mask)
		output = self.linear(output)
		
		return output


class PositionalEncoding_foo(nn.Module):

	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 25):
		super().__init__()
		# Next line added 5.4.2024, Not Tested! ??/ttehas
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe = torch.zeros(1, max_len, d_model).to(self.device)
		pe[0, :, 0::2] = torch.sin(position * div_term)
		pe[0, :, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x: Tensor) -> Tensor:
		"""
		Arguments:
			x: Tensor, shape ``[batch_size, seq_len, input_dim]``
			## x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
		"""
		x = x + self.pe[:x.size(1)]
		return self.dropout(x)


		# Original code:
		
		# pe = torch.zeros(max_len, 1, d_model)
		# pe[:, 0, 0::2] = torch.sin(position * div_term)
		# pe[:, 0, 1::2] = torch.cos(position * div_term)
		# self.register_buffer('pe', pe)

	# def forward(self, x: Tensor) -> Tensor:
		# """
		# Arguments:
			# x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
		# """
		# x = x + self.pe[:x.size(0)]
		# return self.dropout(x)




# THE FOLLOWING TWO FUNCTIONS: writeStatsFile() AND readStatsFile()
# HAVE BEEN COPIED FROM GEN_UTILS.PY (to avoid installing fiona & gdal).

# This function writes the model data statistics (mean & stdev)
# into the specified text file. The mean and std of a certaing 
# data set will be written into one row of the output file
# in the format:
#
# dataID = [mean] [std]
#
# the data items [mean] and [std] may be single numbers, or
# a series of numbers separated with space (i.e. all mean
# values are occupying the first half of each line data values).

def writeStatsFile(dataStats, statsFile):

#statsFile = os.path.join(modelPath, 'inputStats.txt')

	with open(statsFile, 'w') as f: 
		for key, value in dataStats.items():
			dataMean, dataStd = dataStats[key]
			f.write('%s = ' % (key))
			dataMean = dataMean.tolist()
			dataStd = dataStd.tolist()
			if isinstance(dataMean, list):
				for i in range(len(dataMean)):
					f.write('%f ' % (dataMean[i]))
				for i in range(len(dataMean)):
					f.write('%f ' % (dataStd[i]))
				f.write('\n')
			else:
				f.write('%f %f\n' % (dataMean, dataStd))



def readStatsFile(statsFile):

	dataSets = ['s2_data_b2', 's2_data_b3', 's2_data_b4', 's2_data_b5', 's2_data_b8a', 's2_data_b11', 
			   'dem', 'slope_ew', 'slope_ns', 'midpixeldata', 'metadata', 
			   'weather_lepo', 'weather_puuma', 'weather_viro']

	list_variables = []
	params = Params(list_variables)
	params.readFile(statsFile)
	#print(params)

	dataStats = OrderedDict()

	for i, key in enumerate(dataSets):
		print("key = ", key)
		thisKeyData = params.getFloatList(key)
		if thisKeyData is not None:
			datalength = len(thisKeyData)
			print("datalength = ", datalength)
			thisKeyData_np = np.array(thisKeyData)
			dataMean = thisKeyData_np[0:int(datalength/2)]
			dataStd = thisKeyData_np[int(datalength/2):]
			dataStats[key] = (dataMean, dataStd)
	'''	
	print("")
	print(params.getFloatList('s2_data_b2'))
	print(params.getFloatList('midpixeldata'))
	print(np.array(params.getFloatList('s2_data_b2')))
	print(np.array(params.getFloatList('metadata')))
	'''

	return dataStats, params



		
		
'''
# Define the combined and the decoder modules:

# The encoder input dimension is the nbr of weether variables x bi-monthly aggregates
# input_dim_enc = 5 x 24 = 120
input_dim_enc = 120
hid_dim_enc = 128
n_layers_enc = 1
dropout_enc = 0.5
rnn_type = 'GRU'

enc_params = {
	'input_dim_enc': input_dim_enc,
	'hid_dim_enc': hid_dim_enc,
	'n_layers_enc': n_layers_enc,
	'dropout_enc': dropout_enc,
	'rnn_type': rnn_type
	}

# The input dimension of the fully connected block is 
#
# siteInfo parameters: 10
# forest variables: 15
inp_dim_fc = 25
# Number of fully connected MLP hidden layers (including output layer):
nr_hid_fc = 2
hid_sizes_fc = [32, 32]
dropout_fc = 0.5

fc_params = {
	'inp_dim_fc': inp_dim_fc,
	'nr_hid_fc': nr_hid_fc,
	'hid_sizes_fc': hid_sizes_fc,
	'dropout_fc': dropout_fc
	}

# When makin model for one variable at a time, the output dimension is 1
output_dim_dec = 1
# The input dimension is the same as the output dimension (nbr of target variables)
inp_dim_dec = output_dim_dec
hid_dim_dec = hid_dim_enc + int(hid_sizes_fc[-1]/2)
n_layers_dec = n_layers_enc
dropout_dec = 0.5 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("hid_dim_enc = ", hid_dim_enc)
print("hid_dim_dec = ", hid_dim_dec)

comb = Combined(enc_params, fc_params)
dec = Decoder(output_dim_dec, inp_dim_dec, hid_dim_dec, n_layers_dec, dropout_dec, rnn_type=rnn_type)

model = Seq2Seq(comb, dec, device).to(device)


# We initialize weights in PyTorch by creating a function which we apply to our model. 
# When using apply, the init_weights function will be called on every module and sub-module 
# within our model. For each module we loop through all of the parameters and sample them 
# from a uniform distribution with nn.init.uniform_.

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)

'''




# ==================================================================
# GRUModel() Code origin:
#
# https://github.com/SheezaShabbir/Time-series-Analysis-using-LSTM-RNN-and-GRU/blob/main/Pytorch_LSTMs%2CRNN%2CGRU_for_time_series_data.ipynb
#
# ==================================================================

class GRUModel(nn.Module):
	"""GRUModel class extends nn.Module class and works as a constructor for GRUs.

	   GRUModel class initiates a GRU module based on PyTorch's nn.Module class.
	   It has only two methods, namely init() and forward(). While the init()
	   method initiates the model with the given input parameters, the forward()
	   method defines how the forward propagation needs to be calculated.
	   Since PyTorch automatically defines back propagation, there is no need
	   to define back propagation method.

	   Attributes:
		   hidden_dim (int): The number of nodes in each layer
		   layer_dim (str): The number of layers in the network
		   gru (nn.GRU): The GRU model constructed with the input parameters.
		   fc (nn.Linear): The fully connected layer to convert the final state
						   of GRUs to our desired output shape.

	"""
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
		"""The __init__ method that initiates a GRU instance.

		Args:
			input_dim (int): The number of nodes in the input layer
			hidden_dim (int): The number of nodes in each layer
			layer_dim (int): The number of layers in the network
			output_dim (int): The number of nodes in the output layer
			dropout_prob (float): The probability of nodes being dropped out

		"""
		super(GRUModel, self).__init__()

		# Defining the number of layers and the nodes in each layer
		self.layer_dim = layer_dim
		self.hidden_dim = hidden_dim

		# GRU layers
		self.gru = nn.GRU(
			input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
		)

		# Fully connected layer
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		"""The forward method takes input tensor x and does forward propagation

		Args:
			x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

		Returns:
			torch.Tensor: The output tensor of the shape (batch size, output_dim)

		"""
		# Initializing hidden state for first input with zeros
		h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim,device=x.device).requires_grad_()

		# Forward propagation by passing in the input and hidden state into the model
		out, _ = self.gru(x, h0.detach())

		# Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
		# so that it can fit into the fully connected layer
		out = out[:, -1, :]

		# Convert the final state to our desired output shape (batch_size, output_dim)
		out = self.fc(out)

		return out
