import numpy as np
import torch
import time
import math
import os
import datetime
import sys
import pandas as pd

# the function modelEval() replaced with 

#from gen_utils import modelEval
 
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms, utils

import torch.nn as nn
from torcheval.metrics import R2Score

from gen_utils_art import targetDataNorm, joinTestDataWithResults, predsdeNorm
from gen_utils_art import elapsedTime


from sklearn.preprocessing import StandardScaler

#global inputDataScaler
#global targetDataScaler

#inputDataScaler = StandardScaler()
#targetDataScaler = StandardScaler()



# ==========================================================================
# Custom loss function
# --------------------------------------------------------------------------
# targets	(numpy array of shape [C, seq_len, nrVars])
# preds		(numpy array of shape [C, seq_len, nrVars])

class CustomLoss(nn.Module):
	def __init__(self, rmseFactor = 1.0, biasFactor = 2.0, r2limit = 0.25):
		super(CustomLoss, self).__init__()

		self.rmseFactor = rmseFactor
		self.biasFactor = biasFactor
		self.r2limit = r2limit
		
	def forward(self, predictions, targets):
		
		r2metric = R2Score()

		rmse = torch.sqrt(torch.mean(torch.square(predictions - targets)))
		bias = torch.abs(torch.mean(predictions) - torch.mean(targets))
		# Compute r2 score:
		r2metric.update(predictions, targets)
		r2 = torch.max(r2metric.compute(), torch.zeros(1, 1, device=predictions.device))
		
		# Compute loss:
		loss = (self.rmseFactor * rmse + self.biasFactor * bias)/(self.r2limit + r2)

		return loss

# CustomLoss2
#
# This function computes the loss loss = (a * rmse + b * bias)/(c + r2)
# with rmse, bias & r2 values computed as the case-wise means of the batch.
#
# The constants a, b & c are weighing factors:
# a = self.rmseFactor; weight for rmse
# b = self.biasFactor; weight for bias
# c = self.r2limit; limit for r2 to prevent the denominator going to zero

class CustomLoss_perCase(nn.Module):
	def __init__(self, rmseFactor = 1.0, biasFactor = 2.0, r2limit = 0.25):
		super(CustomLoss_perCase, self).__init__()

		self.rmseFactor = rmseFactor
		self.biasFactor = biasFactor
		self.r2limit = r2limit

	def forward(self, predictions, targets):
		
		r2metric = R2Score(multioutput="raw_values")

		# The shape of predictions and targets = [batch_size, seq_len, nr_variables] 
		nrVariables = targets.shape[2]
		
		# Compute the FoM's per variable and per case and take the means later:
		for ii in range(nrVariables):
			# Compute the RMSE values for this variable cases separately (i.e. the RMSE's of
			# the 25 year predictions for each case): keep the dimension (True), so that the
			# shape of rmse_ii is [batch_size, 1].
			# torch.square(.) computes the element-wise squares of the diffrence <preds - tgts>
			# torch.mean(.) computes the row-wise mean for all the samples in the batch
			# torch.sqrt(.) takes the element-wise squares of the means; shape of rmse_ii = [batch_size, 1]
			rmse_ii = torch.sqrt(torch.mean(torch.square(predictions[:,:,ii] - targets[:,:,ii]),1, keepdim = True))

			# Compute absolute bias for each case (= batch element) as with rmse abive. 
			# Note that the value of <preds_mean - tgts_mean> is not divided by tgts_mean,
			# because the input data is N(0,1) normalized, and thus tgts_mean is close to zero.
			# shape of bias_ii = [batch_size, 1]:
			bias_ii = torch.abs(torch.mean(predictions[:,:,ii],1,True) - torch.mean(targets[:,:,ii],1, keepdim = True))
			
			# Compute r2 score per case. The tgts & preds are first transposed to get the
			# r2 values for each case (as with r2metric = R2Score(multioutput="raw_values")
			# specified above). The resulting 1-dim vector is expanded to shape [batch_size, 1]
			# with the unsqueeze(1) operation:
			r2metric.update(torch.transpose(targets[:,:,ii],0,1), torch.transpose(predictions[:,:,ii],0,1))
			r2_ii = r2metric.compute().unsqueeze(1)
			#print(r2_ii)
		
			if ii == 0:
				rmse = rmse_ii
				bias = bias_ii
				r2 = r2_ii
			else:
				# Concatenate the rmse_ii's & bias_ii of all variables to 
				# rmse & bias in dim = 1:
				# shape of rmse/bias = [batch_size, nrVariables]
				rmse = torch.cat((rmse, rmse_ii), dim=1)
				bias = torch.cat((bias, bias_ii), dim=1)
				r2 = torch.cat((r2, r2_ii), dim=1)
		
		# Take the mean of all the elements of rmse, bias & r2. 
		# Limit the minimum r2 value to zero:
		rmse = torch.mean(rmse)
		bias = torch.mean(bias)
		r2 = torch.max(torch.mean(r2), torch.zeros(1, 1, device=predictions.device))
		
		# Compute loss:
		loss = (self.rmseFactor * rmse + self.biasFactor * bias)/(self.r2limit + r2)
		
		return loss


# CustomLoss_perYear
#
# This function computes the loss as loss = (a * rmse + b * bias)/(c + r2)
# with rmse, bias & r2 values computed as the year-wise means of the batch.
#
# The constants a, b & c are weighing factors:
# a = self.rmseFactor; weight for rmse
# b = self.biasFactor; weight for bias
# c = self.r2limit; limit for r2 to prevent the denominator going to zero

class CustomLoss_perYear(nn.Module):
	def __init__(self, rmseFactor = 1.0, biasFactor = 2.0, r2limit = 0.25):
		super(CustomLoss_perYear, self).__init__()

		self.rmseFactor = rmseFactor
		self.biasFactor = biasFactor
		self.r2limit = r2limit

	def forward(self, predictions, targets):
		
		r2metric = R2Score(multioutput="raw_values")

		# The shape of predictions and targets = [batch_size, seq_len, nr_variables] 
		nrVariables = targets.shape[2]
		
		# Compute the FoM's per variable and per case and take the means later:
		for ii in range(nrVariables):
			# Compute the RMSE values for this variable cases separately (i.e. the RMSE's of
			# the 25 year predictions for each year): keep the dimension (True), so that the
			# shape of rmse_ii is [1, seq_len].
			# torch.square(.) computes the element-wise squares of the diffrence <preds - tgts>
			# torch.mean(.) computes the mean  per column for all the samples in the batch (= MS-error)
			# torch.sqrt(.) takes the element-wise squares of the means; shape of rmse_ii = [1, seq_len]
			rmse_ii = torch.sqrt(torch.mean(torch.square(predictions[:,:,ii] - targets[:,:,ii]), dim = 0, keepdim = True))

			# Compute absolute bias for each year as with rmse above. 
			# Note that the value of <preds_mean - tgts_mean> is not divided by tgts_mean,
			# because the input data is N(0,1) normalized, and thus tgts_mean is close to zero.
            # The interpretation of bias_ii is: bias_ii = the yearly absolute values of the biases 
            # computed of the batch samples (sites).
			# shape of bias_ii = [1, seq_len]:
            
			bias_ii = torch.abs(torch.mean(predictions[:,:,ii] - targets[:,:,ii], dim = 0, keepdim = True))
            # The next produces equivalent result wrt. the previous line (i.e. first the column means taken then these means are subtracted from each other)
			#bias_ii = torch.abs(torch.mean(predictions[:,:,ii], dim = 0, keepdim = True) - #torch.mean(targets[:,:,ii], dim = 0, keepdim = True))
			
			# Compute r2 score per year. The tgts & preds are first transposed to get the
			# r2 values for each year (as with r2metric = R2Score(multioutput="raw_values")
			# specified above). The resulting 1-dim vector is expanded to shape [1, seq_len]
			# with the unsqueeze(0) operation:
			r2metric.update(targets[:,:,ii], predictions[:,:,ii])
			r2_ii = r2metric.compute().unsqueeze(0)
			#print(r2_ii)
		
			if ii == 0:
				rmse = rmse_ii
				bias = bias_ii
				r2 = r2_ii
			else:
				# Concatenate the rmse_ii's & bias_ii of all variables to 
				# rmse & bias in dim = 0:
				# shape of rmse/bias = [nrVariables, seq_len]
				rmse = torch.cat((rmse, rmse_ii), dim=0)
				bias = torch.cat((bias, bias_ii), dim=0)
				r2 = torch.cat((r2, r2_ii), dim=0)
		
		# Take the mean of all the elements of rmse, bias & r2. 
		# Limit the minimum r2 value to zero:
		rmse = torch.mean(rmse)
		bias = torch.mean(bias)
		r2 = torch.max(torch.mean(r2), torch.zeros(1, 1, device=predictions.device))
		
		# Compute loss:
		loss = (self.rmseFactor * rmse + self.biasFactor * bias)/(self.r2limit + r2)
		
		return loss


# ==========================================================================
# Art_Dataset_CPU() / Data set definition
# --------------------------------------------------------------------------

class Art_Dataset_CPU(Dataset):
	def __init__(self, dataSetDict, transform=None, target_transform=None):
		
		#global inputDataScaler
		#global targetDataScaler
		
		# ==============================================================================
		# Initialize variables:
		# ------------------------------------------------------------------------------
		prebasDataFile = dataSetDict['prebasDataFile']
		climdataFile = dataSetDict['climdataFile']
		setLabel = dataSetDict['setLabel']
		outPath = dataSetDict['outPath']
		
		# Get data set filters:
		filters_training = dataSetDict['filters_training']
		filters_test = dataSetDict['filters_test']
		
		# Optionally normalize also target data (input data normalized always):
		normalizeTgtData = dataSetDict['normalizeTgtData']
		replaceNans = dataSetDict['replaceNans']
		
		# Assign variables to pass on to __getitem__:
		self.verbose = dataSetDict['verbose']

		# The fully connected layer's input variables are the forest vars + site info vars:
		self.inputVarFcCols = dataSetDict['inputVarFcCols']
		self.targetVars = dataSetDict['targetVars']
		self.targetVarCols = dataSetDict['targetVarCols']
		self.metaDataCols =  dataSetDict['metaDataCols']

		# The next one is for cases when more than one output variable will be predicted
		# at the sam model (default: output_dim_dec = 1):
		self.output_dim_dec = len(dataSetDict['targetVars'])
		#self.output_dim_dec = dataSetDict['output_dim_dec']
		
		# use the next also to organize the target data dimensions:
		self.nYears = dataSetDict['nYears']
		
		# The encoder input variables are the climate data:
		# dataSetDict['climDataCols'] = ['PAR_mean','TAir_mean','Precip_mean','VPD_mean','CO2_mean']
		self.climDataCols = dataSetDict['climDataCols']
		#print("self.climDataCols = ", self.climDataCols)

		# ==============================================================================
		# Read data from files:
		# ------------------------------------------------------------------------------
		# Read the forest variable data and site info data + the target data:
		self.prebasData = pd.read_csv(prebasDataFile)

		# Read the climate data:
		# These data will be joined with the forest var & siteInfo data in the __getitem__
		# section (Note: The input climate data has been normalized in advance):
		self.climData = pd.read_csv(climdataFile)

		# Get climate data column indices for __getitem__:
		self.climdataColIdx = [idx for idx, hdr in enumerate(self.climData.columns) if hdr in self.climDataCols]
		
		# ==============================================================================
		# Apply data filters (i.e. filter input data rows by specifying filter rules):
		# ------------------------------------------------------------------------------
		# NOTE: The first filter must be the string 'setLabel == X', where X = 1, 2 or 3
		# for training, validation and test sets correspondingly. These set filters will
		# be added to the original filter strings in the code below.
		if setLabel < 3:
			filters_training = ['setLabel == ' + str(setLabel)] + filters_training
			for thisFilter in filters_training:
				self.prebasData = self.prebasData.query(thisFilter)
		else:
			filters_test = ['setLabel == ' + str(setLabel)] + filters_test
			for thisFilter in filters_test:
				self.prebasData = self.prebasData.query(thisFilter)

		# ==============================================================================
		# Normalize inputDataFc:
		# ------------------------------------------------------------------------------
		# Compute data set statistics (setLabel=1), or normalize the data (setlabel = 2 or 3):
		# Note: The scikit-learn StandardScalers have been declared as globals:
		inputDataScaler = StandardScaler()
		inputStatsFile = os.path.join(outPath, 'inputStats.txt')
		
		if setLabel == 1:
			# Compute the data normalization statistics with training data set:
			inputDataScaler.fit(self.prebasData[self.inputVarFcCols].values)
			# Save the data stats into text file:
			np.savetxt(inputStatsFile, np.vstack((inputDataScaler.mean_, inputDataScaler.var_, inputDataScaler.scale_)), delimiter=',')
		else:
			# Read the normalization statistics from file for validation and test sets:
			inputDataStats = np.loadtxt(inputStatsFile, delimiter= ',')
			inputDataScaler.mean_ = inputDataStats[0,:]
			inputDataScaler.var_ = inputDataStats[1,:]
			inputDataScaler.scale_ = inputDataStats[2,:]
			
		# Before normalization, replace NaNs with zeros:
		self.inputDataFc = self.prebasData[self.inputVarFcCols].values
		self.inputDataFc = np.nan_to_num(self.inputDataFc, nan=replaceNans)
		
		# Normalize all input data sets with the training data statistics:
		self.inputDataFc = inputDataScaler.transform(self.inputDataFc)

		# ==============================================================================
		# Normalize target data also, if desired (replacement of NaNs is done within
		# targetDataNorm()). Computation of the target data normalization statistics
		# is a little bit more complicated than for input data (all data for nYears)
		# must be flattened first), thus a sub-routine:
		# ------------------------------------------------------------------------------
		if normalizeTgtData:
			tgtStatsFile = os.path.join(outPath, 'targetStats.txt')
			if setLabel == 1:
				# Compute normalization statisics and normalize training data:
				print("Compute target data normalization statisics and normalize training data:")
				self.targetData = targetDataNorm(self.prebasData[self.targetVarCols], self.targetVars,
																   statsFile=tgtStatsFile, mode='fitEtTxform', replaceNans=replaceNans)
				# targetDataScaler, self.targetData = targetDataNorm(self.prebasData[self.targetVarCols], self.targetVars,
																   # mode='fitEtTxform', fileOut=tgtStatsFile, replaceNans=replaceNans)
			else:
				print("Normalize target data with training set statistics:")
				# Normalize target data with training set statistics:
				self.targetData = targetDataNorm(self.prebasData[self.targetVarCols], self.targetVars,  
																   statsFile=tgtStatsFile, mode='transform', replaceNans=replaceNans)
				# targetDataScaler, self.targetData = targetDataNorm(self.prebasData[self.targetVarCols], self.targetVars, targetDataScaler,  
																   # mode='transform', replaceNans=replaceNans)

		# ==============================================================================
		# Extract metadata columns:
		# ------------------------------------------------------------------------------
		self.metaData = self.prebasData[self.metaDataCols]
		
		# ==============================================================================
		# Write the composed (unnormalized) data sets (train, valid & test) to output folder:
		# ------------------------------------------------------------------------------
		if dataSetDict['saveDataSets'] == True:
			saveDataSet = pd.concat([self.metaData, self.prebasData[self.inputVarFcCols], self.prebasData[self.targetVarCols]], axis = 1)
			#saveDataSet = pd.concat([self.prebasData[self.metaDataCols], self.prebasData[self.inputVarFcCols], self.prebasData[self.targetVarCols]], axis = 1)
			strStr = 'train' if setLabel == 1 else 'valid' if setLabel == 2 else 'test'
			dataSetFile = os.path.join(outPath, strStr + 'SetData.csv')
			saveDataSet.to_csv(dataSetFile, sep = ',', index = False)

			# ... and the normalized data as well:
			# cannot concat pd dataframe & numpy arrays!!!
			#saveDataSet = pd.concat([self.metaData, self.inputDataFc, self.targetData], axis = 1)
			##saveDataSet = pd.concat([self.prebasData[self.metaDataCols], self.prebasData[self.inputVarFcCols], self.prebasData[self.targetVarCols]], axis = 1)
			#strStr = 'train' if setLabel == 1 else 'valid' if setLabel == 2 else 'test'
			#dataSetFile = os.path.join(outPath, strStr + 'SetData_norm.csv')
			#saveDataSet.to_csv(dataSetFile, sep = ',', index = False)
		
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.inputDataFc)

	# =================================================================================
	# =================================================================================
	def __getitem__(self, idx):
		#if torch.is_tensor(idx):
		#    idx = idx.tolist()
		
		# Extract metadata for the current (idx) data row:
		thisMetaData = self.metaData.loc[self.metaData.index[idx], :]
		
		# Input data for the fully connected section:
		#thisInputDataFc = self.inputDataFc[idx].astype(np.float32)
		thisInputDataFc = self.inputDataFc[idx, :].astype(np.float32)
		##thisInputDataFc = self.inputDataFc.loc[self.inputDataFc.index[idx], :]
		##thisInputDataFc = thisInputDataFc.values.astype(np.float32)

		# Target data:
		#thisTargetData = self.targetData[idx].astype(np.float32)
		thisTargetData = self.targetData[idx, :].astype(np.float32)
		##thisTargetData = self.targetData.loc[self.targetData.index[idx], :]
		##thisTargetData = thisTargetData.values.astype(np.float32)
		##thisTargetData = thisTargetData.to_numpy(dtype=np.float32)

		# Extract the climate data with of the climate data zone, scenario, and
		# time period indicated by the 'climID_orig', 'scenario', and 'year_start' &
		# 'year_end' parameters of the metadata:
		thisClimID = thisMetaData['climID_orig']
		thisScenario = thisMetaData['scenario']
		thisYear_start = thisMetaData['year_start']
		thisYear_end = thisMetaData['year_end']

		# Extract the corresponding climate data:
		# Construct the filter string for retrieving the climate data (Note, that the years
		# indicated by the variables 'year_start' and 'year_end' in the metadata must both be 
		# included in the range):
		thisClimdata = self.climData
		filters_climData = ['scenario == ' + thisScenario[3:].replace("_", ""), 'climID == ' + str(thisClimID), 'YEAR >= ' + str(thisYear_start), 'YEAR <= ' + str(thisYear_end)]
		for thisFilter in filters_climData:
			thisClimdata = thisClimdata.query(thisFilter)

		# Organize the climate data for the Encoder:
		# The encoder input (climate data) size:
		# enc_inp: [batch_size, seq_len, input_dim]
		for thisYear in range(thisYear_start, thisYear_end+1):
			# Get data for the current year (within period = [thisYear_start, thisYear_end]):
			thisYearFilter = 'YEAR == ' + str(thisYear)
			thisYearData = thisClimdata.query(thisYearFilter)
			# Extract data to numpy array + flatten column-wise:
			#print(thisYearData.iloc[:,self.climdataColIdx])
			thisYearData_np = thisYearData.iloc[:,self.climdataColIdx].values.flatten(order='F')
			#thisYearData_np = thisYearData[self.climDataCols].values.flatten(order='F')

			# the encoder input dimensions are:  [batch_size, seq_len, input_dim] (batch_first = True):
			y = np.expand_dims(thisYearData_np, axis=1)
			#y = np.expand_dims(thisYearData_np, axis=(0,1))   # This adds the batch dimension which is actually added by the data loader (as the first dim)
			if thisYear == thisYear_start:
				encoderInput = y
				#print(encoderInput.shape)
			else:
				# Concatenate in sequence dimension (i.e. year range dim = seq_len):
				# After the for-loop the dimensions will be: dimensions [input_dim, seq_len]
				encoderInput = np.concatenate((encoderInput, y), axis=1)
				#encoderInput = np.concatenate((encoderInput, y), axis=0)

		# Transpose encoderInput to get the dimensions [seq_len, input_dim] (batch_size, added by dataLoader):
		encoderInput = encoderInput.T

		#print(encoderInput.shape)

		# Organize inputDataFc:
		# Input dimensions = [batch size, nbr of features] (= row vector, i.e. 'thisInputDataFc' is ok!):

		# Organize targetData; Target dimensions: [trg_len, batch_size, trg_dim].
		# So far the target data is organized as a row vector, with elements representing
		# consequential years (nYears = 25).

		# Organize the target data first in a two-dimensional Numpy array with dim =  [trg_dim, trg_len]:
		thisTargetData = thisTargetData.reshape(self.output_dim_dec,self.nYears)
		# Then transpose to switch the dimensions to dim = [trg_len, trg_dim]
		thisTargetData = thisTargetData.T

		# Then add the batch dimension (second dim; axis = 1) to have dim = [trg_len, batch_size, trg_dim]
		targetData = thisTargetData
		#targetData = np.expand_dims(thisTargetData, axis=1)
		#targetData = np.expand_dims(thisTargetData, axis=1)

		#print("idx = ", idx)
		#print("thisInputDataFc.shape = ", thisInputDataFc.shape)
		#print("encoderInput.shape = ", encoderInput.shape)
		#print("thisTargetData.shape = ", thisTargetData.shape)
		
		dataItem = {
			'inputDataEnc': encoderInput,
			'inputDataFc': thisInputDataFc,
			'targetData': targetData
		}
			
		if self.transform:
			dataItem = self.transform(dataItem)
		
		return dataItem


class Art_Dataset(Dataset):
	def __init__(self, dataSetDict, transform=None, target_transform=None):
		
		#global inputDataScaler
		#global targetDataScaler
		
		# ==============================================================================
		# Initialize variables:
		# ------------------------------------------------------------------------------
		prebasDataFile = dataSetDict['prebasDataFile']
		climdataFile = dataSetDict['climdataFile']
		setLabel = dataSetDict['setLabel']
		outPath = dataSetDict['outPath']
		
		# Get data set filters:
		filters_training = dataSetDict['filters_training']
		filters_test = dataSetDict['filters_test']
		
		# Optionally normalize also target data (input data normalized always):
		normalizeTgtData = dataSetDict['normalizeTgtData']
		replaceNans = dataSetDict['replaceNans']
		
		# Assign variables to pass on to __getitem__:
		self.verbose = dataSetDict['verbose']

		# The fully connected layer's input variables are the forest vars + site info vars:
		self.inputVarFcCols = dataSetDict['inputVarFcCols']
		self.targetVars = dataSetDict['targetVars']
		self.targetVarCols = dataSetDict['targetVarCols']
		self.metaDataCols =  dataSetDict['metaDataCols']
		self.cascadeTgtVars =  dataSetDict['cascadeTgtVars']
		self.cascadeInputVars = dataSetDict['cascadeInputVars']

		# The next one is for cases when more than one output variable will be predicted
		# at the sam model (default: output_dim_dec = 1):
		self.output_dim_dec = len(dataSetDict['targetVars'])
		#self.output_dim_dec = dataSetDict['output_dim_dec']
		
		# use the next also to organize the target data dimensions:
		self.nYears = dataSetDict['nYears']
		
		# The encoder input variables are the climate data:
		# dataSetDict['climDataCols'] = ['PAR_mean','TAir_mean','Precip_mean','VPD_mean','CO2_mean']
		self.climDataCols = dataSetDict['climDataCols']
		#print("self.climDataCols = ", self.climDataCols)

		# ==============================================================================
		# If the task is to produce a cascade model, then the training, validation, and
		# test data sets will be read from separate, earlier saved *.csv files
		# (trainSetData.csv, validSetData.csv & testSetData.csv). The input parameter
		# dataSetDict['prebasDataFile'] (= paramDict['prebasDataFile']) must contain
		# the proper path to these three files:
		# ------------------------------------------------------------------------------
		if dataSetDict['cascadeInputVars'] is not None:
			prebasFilePath = os.path.dirname(prebasDataFile)
			strStr = 'train' if setLabel == 1 else 'valid' if setLabel == 2 else 'test'
			# The input data set filename (the hard coding to be changed!):
			prebasDataFile = os.path.join(prebasFilePath, strStr + 'SetData_wPredictions.csv')
			
		# Read forest variable data, site info data + the target data from file:
		self.prebasData = pd.read_csv(prebasDataFile)

		# Read the climate data:
		# These data will be joined with the forest var & siteInfo data in the __getitem__
		# section (Note: The input climate data has been normalized in advance):
		self.climData = pd.read_csv(climdataFile)

		# Get climate data column indices for __getitem__:
		self.climdataColIdx = [idx for idx, hdr in enumerate(self.climData.columns) if hdr in self.climDataCols]
		
		# ==============================================================================
		# Apply data filters (i.e. filter input data rows by specifying filter rules):
		# ------------------------------------------------------------------------------
		# NOTE: The first filter must be the string 'setLabel == X', where X = 1, 2 or 3
		# for training, validation and test sets correspondingly. These set filters will
		# be added to the original filter strings in the code below.
		if setLabel < 3:
			filters_training = ['setLabel == ' + str(setLabel)] + filters_training
			for thisFilter in filters_training:
				self.prebasData = self.prebasData.query(thisFilter)
		else:
			filters_test = ['setLabel == ' + str(setLabel)] + filters_test
			for thisFilter in filters_test:
				self.prebasData = self.prebasData.query(thisFilter)

		# ==============================================================================
		# Normalize inputDataFc:
		# ------------------------------------------------------------------------------
		# Compute data set statistics (setLabel=1), or normalize the data (setlabel = 2 or 3):
		# Note: The scikit-learn StandardScalers have been declared as globals:
		inputDataScaler = StandardScaler()
		inputStatsFile = os.path.join(outPath, 'inputStats.txt')
		
		if setLabel == 1:
			# Compute the data normalization statistics with training data set:
			inputDataScaler.fit(self.prebasData[self.inputVarFcCols].values)
			# Save the data stats into text file:
			np.savetxt(inputStatsFile, np.vstack((inputDataScaler.mean_, inputDataScaler.var_, inputDataScaler.scale_)), delimiter=',')
		else:
			# Read the normalization statistics from file for validation and test sets:
			inputDataStats = np.loadtxt(inputStatsFile, delimiter= ',')
			inputDataScaler.mean_ = inputDataStats[0,:]
			inputDataScaler.var_ = inputDataStats[1,:]
			inputDataScaler.scale_ = inputDataStats[2,:]
			
		# Before normalization, replace NaNs with zeros:
		self.inputDataFc = self.prebasData[self.inputVarFcCols].values
		self.inputDataFc = np.nan_to_num(self.inputDataFc, nan=replaceNans)
		
		# Normalize all input data sets with the training data statistics:
		self.inputDataFc = inputDataScaler.transform(self.inputDataFc)
		
		# ==============================================================================
		# Normalize target data also, if desired (replacement of NaNs is done within
		# targetDataNorm()). Computation of the target data normalization statistics
		# is a little bit more complicated than for input data (all data for nYears)
		# must be flattened first), thus a sub-routine:
		# ------------------------------------------------------------------------------
		if normalizeTgtData:
			tgtStatsFile = os.path.join(outPath, 'targetStats.txt')
			if setLabel == 1:
				# Compute normalization statisics and normalize training data:
				print("Compute target data normalization statisics and normalize training data:")
				self.targetData = targetDataNorm(self.prebasData[self.targetVarCols], self.targetVars,
																   statsFile=tgtStatsFile, mode='fitEtTxform', replaceNans=replaceNans)
				# targetDataScaler, self.targetData = targetDataNorm(self.prebasData[self.targetVarCols], self.targetVars,
																   # mode='fitEtTxform', fileOut=tgtStatsFile, replaceNans=replaceNans)
			else:
				print("Normalize target data with training set statistics:")
				# Normalize target data with training set statistics:
				self.targetData = targetDataNorm(self.prebasData[self.targetVarCols], self.targetVars,  
																   statsFile=tgtStatsFile, mode='transform', replaceNans=replaceNans)
				# targetDataScaler, self.targetData = targetDataNorm(self.prebasData[self.targetVarCols], self.targetVars, targetDataScaler,  
																   # mode='transform', replaceNans=replaceNans)

		# ==============================================================================
		# Extract metadata columns:
		# ------------------------------------------------------------------------------
		self.metaData = self.prebasData[self.metaDataCols]
		
		# ==============================================================================
		# Extract cascade target variable columns, if given. This is for saving these
		# variables to the training, validation and test set *.csv files:
		# ------------------------------------------------------------------------------
		if self.cascadeTgtVars is not None:
			self.cascadeTgtVarData = self.prebasData[self.cascadeTgtVars]
		
		# If the 'cascadeInputVars' have been defined, then the current model id a sub-
		# model of an erlier trained model, and uses its' output predictions as model
		# inputs. These inputs - as being time series predictions for nYears -  will 
		# be concatenated with the other time series inputs (= climate variables) to
		# serve as additional inputs the the encoder:
		# ------------------------------------------------------------------------------
		# Normalize cascadeInputVarData also, if they exist:
		# ------------------------------------------------------------------------------
		# Compute data set statistics (setLabel=1), and normalize the data (setlabel = 1, 2 or 3):
		
		if self.cascadeInputVars is not None:
			cascadeInpDataScaler = StandardScaler()
			cascadeInpStatsFile = os.path.join(outPath, 'cascadeInpStats.txt')
			
			if setLabel == 1:
				# Compute the data normalization statistics with training data set:
				cascadeInpDataScaler.fit(self.prebasData[self.cascadeInputVars].values)
				# Save the data stats into text file:
				np.savetxt(cascadeInpStatsFile, np.vstack((cascadeInpDataScaler.mean_, cascadeInpDataScaler.var_, cascadeInpDataScaler.scale_)), delimiter=',')
			else:
				# Read the normalization statistics from file for validation and test sets:
				cascadeInpDataStats = np.loadtxt(cascadeInpStatsFile, delimiter= ',')
				cascadeInpDataScaler.mean_ = cascadeInpDataStats[0,:]
				cascadeInpDataScaler.var_ = cascadeInpDataStats[1,:]
				cascadeInpDataScaler.scale_ = cascadeInpDataStats[2,:]
				
			# Before normalization, replace NaNs with zeros:
			self.cascadeInputVarData = self.prebasData[self.cascadeInputVars].values
			self.cascadeInputVarData = np.nan_to_num(self.cascadeInputVarData, nan=replaceNans)
			
			# Normalize all input data sets with the training data statistics:
			self.cascadeInputVarData = cascadeInpDataScaler.transform(self.cascadeInputVarData)
		
		# ==============================================================================
		# Add here the extration of climate data (dataFrame) corresponding to the
		# filtered prebasData. The returned numpy array is thre-dimensional  with
		# dimensions: [seq_len, input_dim_enc, inputDataSiz], where inputDataSiz =
		# self.metaData.shape[0] (i.e. the data set size):
		# ------------------------------------------------------------------------------
		
		self.climateData = extractClimateData(dataSetDict, self.metaData, self.climData, self.climDataCols)
		#print(self.climateData[:,:,0])

		# ==============================================================================
		# Write the composed (unnormalized) data sets (train, valid & test) to output folder:
		# ------------------------------------------------------------------------------
		if dataSetDict['saveDataSets'] == True:
			if self.cascadeTgtVars is not None:
				saveDataSet = pd.concat([self.metaData, self.prebasData[self.inputVarFcCols], self.prebasData[self.targetVarCols], self.cascadeTgtVarData], axis = 1)
			else:
				saveDataSet = pd.concat([self.metaData, self.prebasData[self.inputVarFcCols], self.prebasData[self.targetVarCols]], axis = 1)

			strStr = 'train' if setLabel == 1 else 'valid' if setLabel == 2 else 'test'
			dataSetFile = os.path.join(outPath, strStr + 'SetData.csv')
			saveDataSet.to_csv(dataSetFile, sep = ',', index = False)

		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.inputDataFc)

	# =================================================================================
	# =================================================================================
	def __getitem__(self, idx):
		#if torch.is_tensor(idx):
		#    idx = idx.tolist()
		
		# Input data for the fully connected section:
		thisInputDataFc = self.inputDataFc[idx, :].astype(np.float32)

		# Target data. So far the target data is organized as a row vector, 
		# with elements representing consequential years (nYears = 25).
		thisTargetData = self.targetData[idx, :].astype(np.float32)

		# Retrieve the cilmate data corresponding to this data item from the 3-D array:
		# Extracted data dimensions: [seq_len, input_dim_enc]
		encoderInput = self.climateData[:,:,idx].astype(np.float32)

		if self.cascadeInputVars is not None:
			# Reshape the cascade input variable data (X variables) into a two-dimensional
			# array of shape (self.nYears, X), and concatenate with climate data:
			cascadeVarData = np.reshape(self.cascadeInputVarData[idx, :], (self.nYears,-1), order='F').astype(np.float32)
			encoderInput = np.hstack((encoderInput, cascadeVarData)).astype(np.float32)

		# Organize targetData; Target dimensions: [trg_len, batch_size, trg_dim].
		# Organize the target data first in a two-dimensional Numpy array with 
		# dim =  [trg_dim, trg_len]:
		thisTargetData = thisTargetData.reshape(self.output_dim_dec,self.nYears)
		# Then transpose to switch the dimensions to dim = [trg_len, trg_dim]
		thisTargetData = thisTargetData.T

		# Assign thisTargetData to targetData directly to have dim = [batch_size, trg_len, trg_dim]
		targetData = thisTargetData
		# Then add the batch dimension (second dim; axis = 1) to have dim = [trg_len, batch_size, trg_dim]
		#targetData = np.expand_dims(thisTargetData, axis=1)

		dataItem = {
			'inputDataEnc': encoderInput,
			'inputDataFc': thisInputDataFc,
			'targetData': targetData
		}
			
		if self.transform:
			dataItem = self.transform(dataItem)
		
		return dataItem




# ==========================================================================
# extractClimateData()
#
# dataSetDict		(dict) The parameter dictionary (= practically paramDict)
# metaData			(pd dataFrame) The data set metadata
# climData			(pd dataFrame) The climate data read from file
# climDataCols		(list of strings) The climate data variables defined in 
#					paramDict['climDataCols']
# --------------------------------------------------------------------------

def extractClimateData(dataSetDict, metaData, climData, climDataCols, verbose = False):

	inputDataSiz = metaData.shape[0]
	
	if verbose:
		print("inputDataSiz = ", inputDataSiz)
		print("climData.shape = ", climData.shape)
		print(climData.head())

	# save the climate data items (corresponding to one site data) into  athre-dimensional
	# numpy array with dimensions [seq_len, input_dim, inputDataSiz]
	climateData = np.ndarray(shape=(dataSetDict['nYears'], dataSetDict['input_dim_enc'], inputDataSiz), dtype = float)
	if verbose:
		print("climateData.shape = ", climateData.shape)

	# Get climate data column indices of the used climate variables:
	climdataColIdx = [ii for ii, hdr in enumerate(climData.columns) if hdr in climDataCols]

	for idx in range(inputDataSiz):
	
		# Extract metadata for the current (idx) data row:
		thisMetaData = metaData.loc[metaData.index[idx], :]

		# Extract the climate data with of the climate data zone, scenario, and
		# time period indicated by the 'climID_orig', 'scenario', and 'year_start' &
		# 'year_end' parameters of the metadata:
		thisClimID = thisMetaData['climID_orig']
		thisScenario = thisMetaData['scenario']
		thisYear_start = thisMetaData['year_start']
		thisYear_end = thisMetaData['year_end']
		
		# if verbose:
			# if idx == 0:
				# print("thisClimID = ", thisClimID)
				# print("thisScenario = ", thisScenario)
				# print("thisYear_start = ", thisYear_start)
				# print("thisYear_end = ", thisYear_end)
		
		# Extract the corresponding climate data:
		# Construct the filter string for retrieving the climate data (Note, that the years
		# indicated by the variables 'year_start' and 'year_end' in the metadata must both be 
		# included in the range):
		
		# Reset climdata to the wholde climate data set here:
		thisClimdata = climData
		
		filters_climData = ['scenario == ' + thisScenario[3:].replace("_", ""), 'climID == ' + str(thisClimID), 'YEAR >= ' + str(thisYear_start), 'YEAR <= ' + str(thisYear_end)]

		for thisFilter in filters_climData:
			# if verbose and idx == 0:
				# print("thisFilter = ", thisFilter)
			thisClimdata = thisClimdata.query(thisFilter)

		# if verbose and idx == 0:
			# print("thisClimdata.shape = ", thisClimdata.shape)
			
		# Organize the climate data for the Encoder:
		# The encoder input (climate data) size:
		# enc_inp: [batch_size, seq_len, input_dim]
		for thisYear in range(thisYear_start, thisYear_end+1):
			# Get data for the current year (within period = [thisYear_start, thisYear_end]):
			thisYearFilter = 'YEAR == ' + str(thisYear)
			thisYearData = thisClimdata.query(thisYearFilter)
			# Extract data to numpy array + flatten column-wise:
			#print(thisYearData.iloc[:,climdataColIdx])
			thisYearData_np = thisYearData.iloc[:,climdataColIdx].values.flatten(order='F')
			#thisYearData_np = thisYearData[climDataCols].values.flatten(order='F')

			# if verbose and idx == 0 and thisYear == thisYear_start:
				# print("thisYearData_np.shape = ", thisYearData_np.shape)
			
			# the encoder input dimensions are:  [batch_size, seq_len, input_dim] (batch_first = True):
			y = np.expand_dims(thisYearData_np, axis=1)
			#y = np.expand_dims(thisYearData_np, axis=(0,1))   # This adds the batch dimension which is actually added by the data loader (as the first dim)
			if thisYear == thisYear_start:
				encoderInput = y
				#print(encoderInput.shape)
			else:
				# Concatenate in sequence dimension (i.e. year range dim = seq_len):
				# After the for-loop the dimensions will be: dimensions [input_dim, seq_len]
				encoderInput = np.concatenate((encoderInput, y), axis=1)
				#encoderInput = np.concatenate((encoderInput, y), axis=0)

		# Transpose encoderInput to get the dimensions [seq_len, input_dim] (batch_size, added by dataLoader):
		encoderInput = encoderInput.T
		climateData[:,:,idx] = encoderInput
		#encoderInput = np.expand_dims(encoderInput, axis=2)
		#climateData[:,:,idx] = encoderInput[:,:,0]
		
		# if verbose and idx == 1:
			# print("encoderInput.shape = ", encoderInput.shape)
			# print(climateData[:,:,idx])
		
	return climateData




    
# ==========================================================================
# Art_Dataset_CPU() / Totensor transform
# --------------------------------------------------------------------------
class ToTensor(object):
	def __init__(self):
		super(ToTensor, self).__init__()
		
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def __call__(self, dataItem):
		
	   return {
			'inputDataEnc': torch.from_numpy(dataItem['inputDataEnc']).float().to(self.device),
			'inputDataFc': torch.from_numpy(dataItem['inputDataFc']).float().to(self.device),
			'targetData': torch.from_numpy(dataItem['targetData']).float().to(self.device)
			}      


# ==========================================================================
# constructDataSets()
#
# This routine calls the function create_datasets() for generating data
# sets (train, valid & test).
# --------------------------------------------------------------------------
def constructDataSets(paramDict, setList = ['train', 'valid', 'test'], shuffles = [True, True, False], verbose = True):

	# paramDict contains most of the variables needed in Art_Dataset_CPU:
	dataSetDict = paramDict

	if 'train' in setList:
		print("dataset_train: ")
		dataSetDict['setLabel'] = 1
		dataset_train = Art_Dataset(dataSetDict=dataSetDict, transform=transforms.Compose([ToTensor()]))
		print("len(dataset_train) = ", len(dataset_train))
		print("")
	else:
		dataset_train = None
		
	if 'valid' in setList:
		print("dataset_valid: ")
		dataSetDict['setLabel'] = 2
		dataset_valid = Art_Dataset(dataSetDict=dataSetDict, transform=transforms.Compose([ToTensor()]))
		print("len(dataset_valid) = ", len(dataset_valid))
		print("")
	else:
		dataset_valid = None

	if 'test' in setList:
		print("dataset_test: ")
		dataSetDict['setLabel'] = 3
		dataset_test = Art_Dataset(dataSetDict=dataSetDict, transform=transforms.Compose([ToTensor()]))
		print("len(dataset_test) = ", len(dataset_test))
		print("")
	else:
		dataset_test = None

	# Construct dataloaders:
	training_loader, validation_loader, test_loader = create_datasets(paramDict['batchSize'], dataset_train, dataset_valid, dataset_test, drop_last=True, shuffles = shuffles)

	return training_loader, validation_loader, test_loader, dataset_train, dataset_valid, dataset_test



# ==========================================================================
# Create dataset loaders
# --------------------------------------------------------------------------
def create_datasets(batch_size, dataset_train, dataset_valid, dataset_test, drop_last=True, shuffles = [True, True, False]):

	if dataset_train is not None:
		training_loader = DataLoader(dataset_train, batch_size=batch_size,
								drop_last=drop_last, shuffle=shuffles[0], num_workers=0)
	else:
		training_loader = None
		
	if dataset_valid is not None:
		validation_loader = DataLoader(dataset_valid, batch_size=batch_size,
								drop_last=drop_last, shuffle=shuffles[1], num_workers=0)
	else:
		validation_loader = None

	if dataset_test is not None:
		test_loader = DataLoader(dataset_test, batch_size=1,
								drop_last=drop_last, shuffle=shuffles[2], num_workers=0)
	else:
		test_loader = None

	return training_loader, validation_loader, test_loader


# ==========================================================================
# Count model (trainable) parameters
# --------------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================================================
# init_weights
# --------------------------------------------------------------------------
# We initialize weights in PyTorch by creating a function which we apply to our model. 
# When using apply, the init_weights function will be called on every module and sub-module 
# within our model. For each module we loop through all of the parameters and sample them 
# from a uniform distribution with nn.init.uniform_.
		
def init_weights(m):
	for name, param in m.named_parameters():
	   nn.init.uniform_(param.data, -0.05, 0.05)
	
	#for name, param in m.named_parameters():
	#    if 'weight' in name and param.data.dim() == 2:
	#        nn.init.kaiming_uniform_(param)



# ==========================================================================
# Early stopping routine
# --------------------------------------------------------------------------
class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 7
			verbose (bool): If True, prints a message for each validation loss improvement. 
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
							Default: 0
			path (str): Path for the checkpoint to be saved to.
							Default: 'checkpoint.pt'
			trace_func (function): trace print function.
							Default: print            
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf
		self.delta = delta
		self.path = path
		self.trace_func = trace_func
	def __call__(self, val_loss, model):

		score = -val_loss

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
		elif score < self.best_score + self.delta:
			self.counter += 1
			if self.verbose:
				self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
			self.counter = 0

	def save_checkpoint(self, val_loss, model):
		'''Saves model when validation loss decrease.'''
		if self.verbose:
			self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), self.path)
		self.val_loss_min = val_loss
		

# ==========================================================================
# Model training function
# --------------------------------------------------------------------------
def train(model, iterator, optimizer, criterion, trainParams, start_time = None):

	modelType = trainParams['modelType']
	clip = trainParams['clip']
	teacher_forcing_ratio = trainParams['teacher_forcing_ratio']
	loss_function = trainParams['loss_function']
	monStep = trainParams['monStep']
	verbose = trainParams['verbose']

	model.train()

	epoch_loss = 0
	#batch_losses = np.zeros((len(iterator),1))

	if start_time is None:
		start_time = time.time()

	for i, dataItem in enumerate(iterator):
		
		inputs_enc = dataItem['inputDataEnc']
		inputs_fc = dataItem['inputDataFc']
		trg = dataItem['targetData']
		
		optimizer.zero_grad()

		if verbose:
			print("train: inputs_enc.shape = ", inputs_enc.shape)
			print("train: inputs_fc.shape = ", inputs_fc.shape)
			print("train: trg.shape = ", trg.shape)
			#print("train: inputs_enc = ", inputs_enc)

		if modelType == 'S2S':
			output = model(inputs_enc, inputs_fc, trg, teacher_forcing_ratio)
		elif modelType == 'FC_RNN':
			output = model(inputs_enc, inputs_fc)
		else:
			# Transformer:
			output = model(inputs_enc, inputs_fc)
			# Cut the extra layers from beginning of the seq_len dimension:
			output = output[:,-trg.shape[1]:]
			
			
		if verbose:
			print("train: output.shape = ", output.shape)
		
		# trg = [batch_size, trg_len, trg_dim]
		# inputs_enc: [batch_size, seq_len, input_dim]
		# inputs_fc: [batch_size, inp_dim_fc]
		# output = [batch_size, trg_len, trg_dim]

		if loss_function == 'CustomLoss_perCase' or loss_function == 'CustomLoss_perYear':
			# This row with CustomLoss_perCase:
			loss = criterion(output, trg)
		else:
			# Organize output & trg for other loss functions: torch.reshape(b, (-1,))
			# The ouput dimension (= trg_dim) is the dimension along the last axis:
			output_dim = output.shape[-1]
			
			# Change 13.3.2024 ??/hja
			output = torch.reshape(output, (-1,output_dim))
			trg = torch.reshape(trg, (-1,output_dim))
			
			#output = output.view(-1, output_dim)
			#trg = trg.view(-1, output_dim)
			
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

			#print("output.shape ", output.shape)
			#print("trg.shape ", trg.shape)
			
			# This row with other loss functions:
			loss = criterion(output, trg)
		
		loss.backward()
		
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		
		optimizer.step()
		
		epoch_loss += loss.item()
		#batch_losses[i] = loss.item()
		
		end_time = time.time()
		t_mins, t_secs = elapsedTime(start_time, end_time)
		
		if i % monStep == 0:
			print("Train/batch: {:02}  loss: {:.3f} | {:02} min {:02} sec \r".format(i, loss.item(), t_mins, t_secs), end="")		
		
		if verbose:
			break
		
	print("                                                                            \r", end="")

	return epoch_loss / len(iterator)


# ==========================================================================
# Model evaluation function
# --------------------------------------------------------------------------
def evaluate(model, iterator, criterion, evalParams, start_time = None):
    
	modelType = evalParams['modelType']
	loss_function = evalParams['loss_function']
	phase = evalParams['phase']
	monStep = evalParams['monStep']
	verbose = evalParams['verbose']
	
	if verbose:
		print("valid: phase = ", phase)

	model.eval()

	epoch_loss = 0
	#batch_losses = np.zeros((len(iterator),1))

	if start_time is None:
		start_time = time.time()
	
	with torch.no_grad():

		for i, dataItem in enumerate(iterator):

			#print('eval: ' + str(i) + '   ', end="")

			inputs_enc = dataItem['inputDataEnc']
			inputs_fc = dataItem['inputDataFc']
			trg = dataItem['targetData']
			
			if i == 0:
				# Use data containers for RMSE computation:
				batch_size = trg.shape[0]
				seq_len = trg.shape[1]
				nrVariables = trg.shape[2]
				targets = np.empty((len(iterator)*batch_size, seq_len, nrVariables), dtype=np.float32)
				preds = np.empty((len(iterator)*batch_size, seq_len, nrVariables), dtype=np.float32)

			if modelType == 'S2S':
				output = model(inputs_enc, inputs_fc, trg, 0) #turn off teacher forcing
			elif modelType == 'FC_RNN':
				output = model(inputs_enc, inputs_fc)
			else:
				# Transformer:
				output = model(inputs_enc, inputs_fc)
				# Cut the extra layers from beginning of the seq_len dimension:
				output = output[:,-trg.shape[1]:]

			#trg = [batch size, trg len]
			#output = [batch size, trg len, output dim]

			# Save targets and predictions into Numpy arrays: 
			#trg_np = trg.detach().numpy()
			#output_np = output.detach().numpy()

			startIdx = i*batch_size
			endIdx = (i+1)*batch_size
			
			# Convert trg & output for cpu (if there not already)
			# for saving (next two lines):
			trg1 = trg.cpu()
			output1 = output.cpu()
			
			targets[startIdx:endIdx,:] = trg1.detach().numpy()
			preds[startIdx:endIdx,:] = output1.detach().numpy()

			if phase == 'valid':
				if loss_function == 'CustomLoss_perCase' or loss_function == 'CustomLoss_perYear':
					# This row with CustomLoss_perCase:
					loss = criterion(output, trg)
				else:
					# Organize output & trg for other loss functions:
					output_dim = output.shape[-1]
					
					# Change 13.3.2024 ??/hja
					output = torch.reshape(output, (-1,output_dim))
					trg = torch.reshape(trg, (-1,output_dim))
				
					#output = output.view(-1, output_dim)
					#trg = trg.view(-1, output_dim)

					#trg = [trg len * batch size]
					#output = [trg len * batch size, output dim]

					# This row with other loss functions:
					loss = criterion(output, trg)
				
				epoch_loss += loss.item()
				#batch_losses[i] = loss.item()

			end_time = time.time()
			t_mins, t_secs = elapsedTime(start_time, end_time)
				
			if i % monStep == 0:
				if phase == 'valid':
					print("Eval/batch: {:02}  loss: {:.3f} | {:02} min {:02} sec \r".format(i, loss.item(), t_mins, t_secs), end="")
				else:
					print("Eval/batch: {:02} | {:02} min {:02} sec \r".format(i, t_mins, t_secs), end="")
			
			#trg_np = trg.detach().numpy()
			#output_np = output.detach().numpy()

	print("                                                                      \r", end="")
		
	return epoch_loss / len(iterator), targets, preds




# ==========================================================================
# Model evaluation function for measuring elapsed time for prediction
# --------------------------------------------------------------------------
def evaluate_silent(model, iterator, criterion, evalParams, start_time = None):
    
	modelType = evalParams['modelType']
	loss_function = evalParams['loss_function']
	phase = evalParams['phase']
	monStep = evalParams['monStep']
	verbose = False
	
	if verbose:
		print("valid: phase = ", phase)

	model.eval()

	epoch_loss = 0
	#batch_losses = np.zeros((len(iterator),1))

	#if start_time is None:
	#	start_time = time.time()
	
	with torch.no_grad():

		for i, dataItem in enumerate(iterator):

			#print('eval: ' + str(i) + '   ', end="")

			inputs_enc = dataItem['inputDataEnc']
			inputs_fc = dataItem['inputDataFc']
			trg = dataItem['targetData']
			
			if i == 0:
				# Use data containers for RMSE computation:
				batch_size = trg.shape[0]
				seq_len = trg.shape[1]
				nrVariables = trg.shape[2]
				targets = np.empty((len(iterator)*batch_size, seq_len, nrVariables), dtype=np.float32)
				preds = np.empty((len(iterator)*batch_size, seq_len, nrVariables), dtype=np.float32)

			if modelType == 'S2S':
				output = model(inputs_enc, inputs_fc, trg, 0) #turn off teacher forcing
			elif modelType == 'FC_RNN':
				output = model(inputs_enc, inputs_fc)
			else:
				# Transformer:
				output = model(inputs_enc, inputs_fc)
				# Cut the extra layers from beginning of the seq_len dimension:
				output = output[:,-trg.shape[1]:]

			#trg = [batch size, trg len]
			#output = [batch size, trg len, output dim]

			# Save targets and predictions into Numpy arrays: 
			#trg_np = trg.detach().numpy()
			#output_np = output.detach().numpy()

			startIdx = i*batch_size
			endIdx = (i+1)*batch_size
			
			# Convert trg & output for cpu (if there not already)
			# for saving (next two lines):
			trg1 = trg.cpu()
			output1 = output.cpu()
			
			targets[startIdx:endIdx,:] = trg1.detach().numpy()
			preds[startIdx:endIdx,:] = output1.detach().numpy()

			if phase == 'valid':
				if loss_function == 'CustomLoss_perCase' or loss_function == 'CustomLoss_perYear':
					# This row with CustomLoss_perCase:
					loss = criterion(output, trg)
				else:
					# Organize output & trg for other loss functions:
					output_dim = output.shape[-1]
					
					# Change 13.3.2024 ??/hja
					output = torch.reshape(output, (-1,output_dim))
					trg = torch.reshape(trg, (-1,output_dim))
				
					#output = output.view(-1, output_dim)
					#trg = trg.view(-1, output_dim)

					#trg = [trg len * batch size]
					#output = [trg len * batch size, output dim]

					# This row with other loss functions:
					loss = criterion(output, trg)
				
				epoch_loss += loss.item()
				#batch_losses[i] = loss.item()

			#end_time = time.time()
			#t_mins, t_secs = elapsedTime(start_time, end_time)
				
			#if i % monStep == 0:
			#	if phase == 'valid':
			#		print("Eval/batch: {:02}  loss: {:.3f} | {:02} min {:02} sec \r".format(i, loss.item(), t_mins, t_secs), end="")
			#	else:
			#		print("Eval/batch: {:02} | {:02} min {:02} sec \r".format(i, t_mins, t_secs), end="")
			
			#trg_np = trg.detach().numpy()
			#output_np = output.detach().numpy()

	#print("                                                                      \r", end="")
		
	return epoch_loss / len(iterator), targets, preds





# ==========================================================================
# Model training loop
# --------------------------------------------------------------------------
def trainingLoop(model, paramDict, training_loader, validation_loader, optimizer, criterion, verbose = False, monStep = 100, monAllEpochs = True):

	N_EPOCHS = paramDict['train_epochs']
	# Define loss % time monitoring step (= nbr of batches):
	#monStep = 100

	trainParams = {
		'modelType': paramDict['modelType'],
		'clip': paramDict['clip_grad'],
		'teacher_forcing_ratio': paramDict['teacher_forcing_ratio'],
		'loss_function': paramDict['loss_function'],
		'monStep': monStep,
		'verbose': verbose
		}

	evalParams = {
		'modelType': paramDict['modelType'],
		'loss_function': paramDict['loss_function'],
		'phase': 'valid',
		'monStep': monStep,
		'verbose': verbose
		}

	modelFile = os.path.join(paramDict['outPath'], paramDict['modelIDout'] + '.pt')

	# initialize the early_stopping object
	early_stopping = EarlyStopping(patience=paramDict['patience'], path=modelFile, delta=paramDict['min_delta'], verbose=False)

	# Collect training & valid loss history:
	trainvalidLosses = pd.DataFrame(np.zeros([N_EPOCHS, 3])*np.nan, index=range(1,N_EPOCHS+1), columns=['Train', 'Valid', 'Time'])

	if monAllEpochs:
		print("Loss/epoch")
		print("Epoch   train    valid")
		print("_______________________")
	
	bestValidLoss = 9999.0
	monTrainLoss = 9999.0
	
	tStart_modelTrain = time.time()

	for epoch in range(N_EPOCHS):
		
		start_time = time.time()
		
		train_loss = train(model, training_loader, optimizer, criterion, trainParams, start_time)
		
		# The verbose mode is only for tensor size checking; return here
		if verbose:
			return model, None
			
		valid_loss, targetsValid, predsValid = evaluate(model, validation_loader, criterion, evalParams, start_time)

		end_time = time.time()
		
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
		
		#print("")
		#print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')

		if monAllEpochs:
			print("{:02}      {:.3f}    {:.3f}  | Time: {:02} min {:02} sec".format(epoch+1, train_loss, valid_loss, epoch_mins, epoch_secs))

		# Update training & valid losses table:
		trainvalidLosses.iloc[epoch,:] = [train_loss, valid_loss, epoch_mins+epoch_secs/60]

		# Drop the optionally remaining NaN rows (does not work within the loop!) from 'trainvalidLosses' & save:
		#trainvalidLosses = trainvalidLosses.dropna()
		lossFile = os.path.join(paramDict['outPath'], 'trainvalidLosses.csv')
		trainvalidLosses.to_csv(lossFile, index=range(epoch))
		
		# Early_stopping checks the validation loss if it has decresed, 
		# and if it has, it will make a checkpoint of the current model:
		early_stopping(valid_loss, model)
		
		# Save the best validation loss and the corresponding training loss:
		if valid_loss < bestValidLoss:
			bestValidLoss = valid_loss
			monTrainLoss = train_loss
		
		if early_stopping.early_stop:
			print("Early stopping")
			break
			
	tEnd_modelTrain = time.time()
	train_mins, train_secs = epoch_time(tStart_modelTrain, tEnd_modelTrain)
		
	print("Minimum validation loss:")	
	print("{:02}   {:.3f}    {:.3f}  | Time: {:02} min {:02} sec".format(epoch+1, monTrainLoss, bestValidLoss, train_mins, train_secs))
	
	# load the last checkpoint with the best model:
	model.load_state_dict(torch.load(modelFile))

	return model, trainvalidLosses, bestValidLoss, monTrainLoss


# ==========================================================================
# trainValidPredictions()
#
# Produce predictions for training & validation data sets + save with inputs
# --------------------------------------------------------------------------
def trainValidPredictions(paramDict, model, criterion, monStep = 10):

	tgtStatsFile = os.path.join(paramDict['outPath'], 'targetStats.txt')

	# Construct data Sets for Producing Predictions. Note that the data set shuffling
	# is set to False for all sets:
	training_loader, validation_loader, foo, dataset_train, dataset_valid, bar = constructDataSets(paramDict, setList = ['train', 'valid'], shuffles = [False, False, False])

	evalParams = {
		'modelType': paramDict['modelType'],
		'loss_function': paramDict['loss_function'],
		'phase': 'test',
		'monStep': monStep,
		'verbose': False
		}

	inputSets = ['training', 'valid']

	# Produce model response for both sets, de-normalize the results, and
	# join the results with the input data sets:
	for thisSet in inputSets:
		if thisSet == 'training':
			dataLoader = training_loader
			inpDataFile = os.path.join(paramDict['outPath'], 'trainSetData.csv')
		else:
			dataLoader = validation_loader
			inpDataFile = os.path.join(paramDict['outPath'], 'validSetData.csv')
		
		# ----------------------------------------------------------------------------------
		# 1. Compute model response for the given set, and de-normalize the results:
		# ----------------------------------------------------------------------------------
		foo, targets, preds = evaluate(model, dataLoader, criterion, evalParams)
		targets_denorm, preds_denorm = predsdeNorm(targets, preds, statsFile = tgtStatsFile)
		
		# ----------------------------------------------------------------------------------
		# 2. Join the results (predictions, and optionally targets) with the input data set: 
		# ----------------------------------------------------------------------------------
		resultsJoined_df = joinTestDataWithResults(paramDict, inpDataFile, preds_denorm, 
												   targets = targets_denorm)

	return None



# ==========================================================================
# searchWrapper()
#
# This function runs the model training with a set of input parameter 
# combinations that have been defined in paramDict. The output for ranking
# the different combination is the best validation loss (bestValidLoss)
# recorded during the model training. The parameter combibations and the
# resulting validation loss will be returned in Pandas DataFrame 'wrapperTbl'
# which will alos be written into a *.csv file, if output file name will be
# provided.
# --------------------------------------------------------------------------

# NOTE: Moved to artisTrainWrapper.py

'''
def searchWrapper_foo(paramDict, training_loader, validation_loader, outFile = None, verbose = False):

	modelTypes = paramDict['modelTypes']
	rnnTypes = paramDict['rnnTypes']
	nrEncLayerss = paramDict['nrEncLayerss']
	encHidDimss = paramDict['encHidDimss']
	learningRates = paramDict['learningRates']
	batchSizes = paramDict['batchSizes']
	encDropouts = paramDict['encDropouts']
	fcDropouts = paramDict['fcDropouts']

	nrTbRows = len(modelTypes)*len(rnnTypes)*len(encHidDimss)*len(nrEncLayerss)*len(learningRates)*len(batchSizes)*len(encDropouts)*len(fcDropouts)

	colHdrs = ['modelType','rnn_type','n_layers_enc','hid_dim_enc','learning_rate','batchSize','dropout_enc','dropout_fc', 'trainLoss', 'validLoss']
	wrapperTbl = pd.DataFrame(index = range(nrTbRows), columns = colHdrs)

	if verbose:
		print("modelTypes = ", modelTypes)
		print("rnnTypes = ", rnnTypes)
		print("nrEncLayerss = ", nrEncLayerss)
		print("encHidDimss = ", encHidDimss)
		print("learningRates = ", learningRates)
		print("batchSizes = ", batchSizes)
		print("encDropouts = ", encDropouts)
		print("fcDropouts = ", fcDropouts)
		print("wrapperTbl.shape = ", wrapperTbl.shape)

	tblCtr = 0

	for thisModelType in modelTypes:
		paramDict['modelType'] = thisModelType

		for thisRnnType in rnnTypes:
			paramDict['rnn_type'] = thisRnnType

			for thisNrEcLayers in nrEncLayerss:
				paramDict['n_layers_enc'] = thisNrEcLayers
				
				for thisEncHidDim in encHidDimss:
					paramDict['hid_dim_enc'] = thisEncHidDim

					for thisLearningRate in learningRates:
						paramDict['learning_rate'] = thisLearningRate

						for thisBatchSize in batchSizes:
							paramDict['batchSize'] = thisBatchSize
							
							for thisEncDropout in encDropouts:
								paramDict['dropout_enc'] = thisEncDropout
		
								for thisFcDropout in fcDropouts:
									paramDict['dropout_fc'] = thisFcDropout
		
									wrapperTbl.iloc[tblCtr,0] = thisModelType
									wrapperTbl.iloc[tblCtr,1] = thisRnnType
									wrapperTbl.iloc[tblCtr,2] = thisNrEcLayers
									wrapperTbl.iloc[tblCtr,3] = thisEncHidDim
									wrapperTbl.iloc[tblCtr,4] = thisLearningRate
									wrapperTbl.iloc[tblCtr,5] = thisBatchSize
									wrapperTbl.iloc[tblCtr,6] = thisEncDropout
									wrapperTbl.iloc[tblCtr,7] = thisFcDropout
		
									# Model definition
									model, criterion, optimizer, scheduler = initModel(paramDict)
									
									# Call trainingLoop() with the defined model and the given parameters:
									model, trainvalidLosses, bestValidLoss, monTrainLoss = trainingLoop(model, paramDict, training_loader, validation_loader, optimizer, criterion, verbose = False, monAllEpochs = True)
		
									wrapperTbl.iloc[tblCtr,8] = monTrainLoss
									wrapperTbl.iloc[tblCtr,9] = bestValidLoss
									tblCtr += 1

	if outFile is not None:
		wrapperTbl.to_csv(outFile, sep = ',', index = False)

	return wrapperTbl
'''








# ==========================================================================
# epoch_time() = a function to show how long an epoch takes.
#
# MOVED TO: gen_utils_art.py
# --------------------------------------------------------------------------
#def elapsedTime(start_time, end_time):
#    elapsed_time = end_time - start_time
#    elapsed_mins = int(elapsed_time / 60)
#    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#    return elapsed_mins, elapsed_secs
	
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



'''
def evaluate(model, iterator, criterion, evalParams):
    
	modelType = evalParams['modelType']
	loss_function = evalParams['loss_function']

	model.eval()

	epoch_loss = 0

	with torch.no_grad():

		for i, dataItem in enumerate(iterator):
			#print('eval: ' + str(i) + '   ', end="")

			inputs_enc = dataItem['inputDataEnc']
			inputs_fc = dataItem['inputDataFc']
			trg = dataItem['targetData']
			
			if modelType == 'S2S':
				output = model(inputs_enc, inputs_fc, trg, 0) #turn off teacher forcing
			else:
				output = model(inputs_enc, inputs_fc)

			#trg = [batch size, trg len]
			#output = [batch size, trg len, output dim]

			if loss_function == 'CustomLoss2':
				# This row with CustomLoss2:
				loss = criterion(output, trg)
			else:
				# Organize output & trg for other loss functions:
				output_dim = output.shape[-1]
				
				output = output.view(-1, output_dim)
				trg = trg.view(-1, output_dim)

				#trg = [trg len * batch size]
				#output = [trg len * batch size, output dim]

				# This row with other loss functions:
				loss = criterion(output, trg)
			
			epoch_loss += loss.item()

			print('eval ' + str(i) + ': ' + str(loss) + '   ', end='')
			print("\r", end="")

			#trg_np = trg.detach().numpy()
			#output_np = output.detach().numpy()

	print("                                                           \r", end="")
		
	return epoch_loss / len(iterator)
'''


# ==========================================================================
# Model training using validation data set and early stopping criterion
# --------------------------------------------------------------------------
#def trainRegrModel(model, dataLoaders, nrOutputs, batch_size, patience, n_epochs):

def trainRegrModel(model, dataLoaders, optimizer, scheduler, lossFunction, trainDict):
    
	# to track the training loss as the model trains
	train_losses = []
	# to track the validation loss as the model trains
	valid_losses = []
	# to track the average training loss per epoch as the model trains
	avg_train_losses = []
	# to track the average validation loss per epoch as the model trains
	avg_valid_losses = [] 

	# Extract data train and valid data loaders:
	train_loader = dataLoaders['train_loader']
	valid_loader = dataLoaders['valid_loader']
	
	n_epochs = trainDict['nrEpochs']
	patience = trainDict['patience']
	nrOutputs = trainDict['nrOutputs']

	# initialize the early_stopping object
	early_stopping = EarlyStopping(patience=patience, verbose=False)

	tic = time.perf_counter()

	print("___________________________________________________________")

	for epoch in range(1, n_epochs + 1):
		targetsTrain = np.zeros((1,nrOutputs), dtype=np.float64)
		predsTrain = np.zeros((1,nrOutputs), dtype=np.float64)
		targetsValid = np.zeros((1,nrOutputs), dtype=np.float64)
		predsValid = np.zeros((1,nrOutputs), dtype=np.float64)
		
		# ================================================================================
		# train the model #
		# ================================================================================
		model.train(True)
		for i, dataItem in enumerate(train_loader):
			#for i, dataItem in enumerate(train_loader_fc):
			# Every data instance is an input + label pair
			inputs = dataItem['inputs']
			targets = dataItem['targets']

			# Forward pass
			outputs = model(inputs)
			#print('inputs = ', inputs)
			#print('outputs = ', outputs)
			#print('targets = ', targets)

			# Zero your gradients for every batch!
			optimizer.zero_grad()
			
			# Compute the loss and its gradients
			loss = lossFunction(outputs, targets)
			loss.backward()
			# Adjust learning weights
			optimizer.step()

			targets_np = targets.detach().numpy()
			outputs_np = outputs.detach().numpy()
			if i < 0:
				print("targets_np.shape = ", targets_np.shape)
				print("outputs_np.shape = ", outputs_np.shape)
			#print("targets_np = ", targets_np)
			#print("outputs_np = ", outputs_np)

			targetsTrain = np.concatenate((targetsTrain, targets_np), axis=0)
			predsTrain = np.concatenate((predsTrain, outputs_np), axis=0)
			#targetsTrain[i*batch_size:(i+1)*batch_size,:] = targets_np
			#predsTrain[i*batch_size:(i+1)*batch_size,:] = outputs_np
			
			# record training loss
			#train_losses.append(loss.item())

		
		# ================================================================================
		# validate the model #
		# ================================================================================
		model.eval() # prep model for evaluation
		for i, dataItem_valid in enumerate(valid_loader):
			inputs_valid = dataItem['inputs']
			targets_valid = dataItem['targets']
			
			outputs_valid = model(inputs_valid)
					
			vloss = lossFunction(outputs_valid, targets_valid)
			# record validation loss
			valid_losses.append(vloss.item())
			
			targets_valid_np = targets_valid.detach().numpy()
			outputs_valid_np = outputs_valid.detach().numpy()
			
			try:
				targetsValid = np.concatenate((targetsValid, targets_valid_np), axis=0)
				predsValid = np.concatenate((predsValid, outputs_valid_np), axis=0)
				if i < 0:
					print("vloss = ", vloss)
					print("targets_valid_np.shape = ", targets_valid_np.shape)
					print("outputs_valid_np.shape = ", outputs_valid_np.shape)
					#print("targets_valid_np = ", targets_valid_np)
					#print("outputs_valid_np = ", outputs_valid_np)
				#targetsValid[i*batch_size:(i+1)*batch_size,:] = targets_valid_np
				#predsValid[i*batch_size:(i+1)*batch_size,:] = outputs_valid_np
			except:
				if i < 0:
					print("targets_valid_np.shape = ", targets_valid_np.shape)
					print("outputs_valid_np.shape = ", outputs_valid_np.shape)
				#print("targets_valid_np = ", targets_valid_np)
				#print("outputs_valid_np = ", outputs_valid_np)
		
		# Adjust learning rate:
		scheduler.step()
		if epoch == 1:
			print("targetsTrain.shape = ", targetsTrain.shape)
			print("predsTrain.shape = ", predsTrain.shape)

			print("targetsValid.shape = ", targetsValid.shape)
			print("predsValid.shape = ", predsValid.shape)
			print("___________________________________________________________")
			print("")
			print("")

			print("Epoch  RMSE_train  /  RMSE_valid  /  Time (hours:mins:secs)")
			print("___________________________________________________________")
		
		# Compute RMSE (train & vaid) over one epoch (remove the dummy first lines):
		foo, FOMs_train = modelEval(targetsTrain[1:,:], predsTrain[1:,:])
		foo, FOMs_valid = modelEval(targetsValid[1:,:], predsValid[1:,:])
		#foo, FOMs_train = modelEval(targetsTrain, predsTrain)
		#foo, FOMs_valid = modelEval(targetsValid, predsValid)
		RMSE_train = FOMs_train["RMSEp_out_mean"]
		RMSE_valid = FOMs_valid["RMSEp_out_mean"]
		
		#FOMs_train = computeFoMs_oneDim(targetsTrain, predsTrain, 'PLOT_V', verbose = 0)
		#FOMs_valid = computeFoMs_oneDim(targetsValid, predsValid, 'PLOT_V', verbose = 0)
		#RMSE_train = FOMs_train["RMSEp_test"]
		#RMSE_valid = FOMs_valid["RMSEp_test"]
		
		# Ensure that the validation error in first epoch is not less than in
		# subsequent epochs:
		#if epoch == 1:
		#    RMSE_valid *= 2
		
		avg_train_losses.append(RMSE_train)
		avg_valid_losses.append(RMSE_valid)
		
		# print training/validation statistics 
		# calculate average loss over an epoch
		#train_loss = np.average(train_losses)
		#valid_loss = np.average(valid_losses)
		#avg_train_losses.append(train_loss)
		#avg_valid_losses.append(valid_loss)
		
		#epoch_len = len(str(n_epochs))
		toc = time.perf_counter()        
		hourss = math.floor((toc-tic)/3600)
		minss = math.floor(((toc-tic)%3600)/60)
		secss = (toc-tic)%60
		
		#print('%0d \t %2.2f %2.2f', epoch, RMSE_train, RMSE_valid)
		print(f"{epoch:2d}:  {RMSE_train:3.1f}  /  {RMSE_valid:3.1f}  /  {hourss:0.0f}:{minss:0.0f}:{secss:0.1f} ")
		#print('{0:4d} \t {1:%2.2f} \t {2:%2.2f}'.format(epoch, RMSE_train, RMSE_valid))
		#print('The value of pi is approximately %5.3f.' % math.pi)
		#print_msg = (f'
		#print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
		#             f'RMSE_train: {RMSE_train} ' +
		#             f'RMSE_valid: {RMSE_valid}')
		
		#print(print_msg)
		
		# clear lists to track next epoch
		#train_losses = []
		#valid_losses = []
		
		# Early_stopping needs the validation loss (RMSE_valid) to check if it has decreased, 
		# and if it has, it will make a checkpoint of the current model
		early_stopping(RMSE_valid, model)
		
		if early_stopping.early_stop:
			print("Early stopping")
			break

	print("")
	# load the last checkpoint with the best model:
	model.load_state_dict(torch.load('checkpoint.pt'))

	return  model, avg_train_losses, avg_valid_losses
	
