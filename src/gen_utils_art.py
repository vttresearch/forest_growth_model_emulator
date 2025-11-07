import os
import datetime
import sys
import math
import random
import time

from datetime import timedelta, date
from datetime import datetime as dt

from param_utils import Params, parseModelParameters, parseModelParams

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

from shutil import copyfile



def bin_continuous_variable(y, nrStrata=10):
    # Bin the continuous variable into discrete categories
    cc, bins = pd.qcut(y, q=nrStrata, retbins=True, labels=False, duplicates='drop')
    
    # Handle NaN values by assigning them to a specific category
    cc = np.where(pd.isna(cc), -1, cc)
    
    return cc, bins


def get_set_labels(y_binned, val_size, test_size):
    set_labels = np.zeros_like(y_binned)
    set_labels[y_binned == -1] = 3  # NaN values
    set_labels[(y_binned >= 0) & (y_binned < val_size * 100)] = 1  # Training set
    set_labels[(y_binned >= val_size * 100) & (y_binned < (val_size + test_size) * 100)] = 2  # Validation set
    set_labels[y_binned >= (val_size + test_size) * 100] = 3  # Test set
    return set_labels


# stratified_split()
#
# This function splits the data set 'dataIn' (= Pandas dataframe) into
# training, validation and test sets using a stratification variable
# (defined by dataIn[stratVarHdr]). The percentage shares of validation
# and test sets will be given as function inputs, the remaining share
# of the input data (1 - val_size - test_size) will be assigned to training 
# set. The fnction uses the 'StratifiedShuffleSplit' algorithm of scikit.learn
# library.
#
# Inputs:
#
# dataIn		(Pnandas DataFrame) A 2D dataframe that contains the input data
#				to be split into three sets (training, validation & test).
#
# stratVarHdr	(string) Header of the stratification variable column. 
#				A continuous-valued stratification variable is expected.
#
# val_size		(float) The size (proportion of original data set size) of the
#				validation set (range = [0, 1]).
#
# test_size		(float) The size (proportion of original data set size) of the
#				test set (range = [0, 1]). The training set size is: 
#				1 - val_size - test_size
#
# nrStrata		(int) The number of strata used in the stratified random sampling.
#				Default: nrStrata=10
#
# outFile		(string) The path of the output file (path + filename). If given,
#				the input data amended with the column 'set_label' will be written
#				as *.csv file to the given location. Default: outFile=None.

def stratified_split(dataIn, stratVarHdr, val_size=0.3, test_size=0.1, nrStrata=10, random_seed=None, outFile=None):

	if isinstance(stratVarHdr, str):
		stratVar = dataIn[stratVarHdr].values
	else:
		print("Stratification variable header not of type str!")
		return None

	stratVar_binned, foo = bin_continuous_variable(stratVar, nrStrata=nrStrata)
	set_labels = get_set_labels(stratVar_binned, val_size, test_size)

	# Combine dataIn, stratVar_binned, and set_labels into a single DataFrame for ease of use
	if isinstance(dataIn, np.ndarray):
		dataIn = pd.DataFrame(dataIn)
	if isinstance(stratVar_binned, np.ndarray):
		stratVar_binned = pd.Series(stratVar_binned, name="stratify_column")
	set_labels = pd.Series(set_labels, name="setLabel")
	df = pd.concat([dataIn, stratVar_binned, set_labels], axis=1)

	# Use StratifiedShuffleSplit to perform stratified splitting based on the binned variable
	sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=random_seed)

	# Stratified Splitting
	for train_index, test_index in sss.split(df, df["stratify_column"]):
		train_set = df.iloc[train_index]
		test_set = df.iloc[test_index]

	# Further split the test set into validation and test sets
	val_set, test_set = train_test_split(test_set, test_size=test_size / (val_size + test_size),
										 stratify=test_set["stratify_column"], random_state=random_seed)

	# Extract X, y, and set_labels from the split sets

	X_train = train_set.drop(["stratify_column"], axis=1)
	X_val = val_set.drop(["stratify_column"], axis=1)
	X_test = test_set.drop(["stratify_column"], axis=1)
	# X_train, y_train, set_labels_train = train_set.drop(["stratify_column"], axis=1), train_set["stratify_column"], train_set["set_label"]
	# X_val, y_val, set_labels_val = val_set.drop(["stratify_column"], axis=1), val_set["stratify_column"], val_set["set_label"]
	# X_test, y_test, set_labels_test = test_set.drop(["stratify_column"], axis=1), test_set["stratify_column"], test_set["set_label"]

	# Change the setlabels for validation and test sets (valid = 2, test = 3):
	X_val['setLabel'] = X_val['setLabel'] * 2
	X_test['setLabel'] = X_test['setLabel'] * 3

	X_out = pd.concat([X_train, X_val, X_test], ignore_index=True)

	if outFile is not None:
		X_out.to_csv(outFile, index=False)

	return X_out



def computeResults(targets, predictions, targetVars, FOMs = ['RMSE', 'RMSEp', 'BIAS', 'BIASp', 'R2', 'Ymean'], exstFile = None, outPath = None):

	speciess = ['pine', 'spr', 'bl']

	nrCases = targets.shape[0]
	seq_len = targets.shape[1]
	nrVariables = targets.shape[2]
	nrFoms = len(FOMs)

	# Extract the species presence/absence table from the saved test data 
	# set file. The table contains indicator if the species at hand was
	# present (= 1) or absent (= 0) at the site (stand):
	if exstFile is not None:
		exstTbl = pd.read_csv(exstFile)
		# Get also the siteID from exstFile, if given:
		siteID = exstTbl['siteID'].values
		exstTbl = exstTbl[['exst_pine', 'exst_spr', 'exst_bl']].values
	else:
		siteID = None
		
	fomsFlatTbl = pd.DataFrame(columns=FOMs, index=targetVars)
	#fomsFlatTbl = pd.DataFrame(index=FOMs)
	fomsFlat = np.zeros((nrVariables, nrFoms))

	#fomsPerYearDict = OrderedDict()
	fomsPerYearDict = {}
	fomsPerYear = np.zeros((seq_len, nrFoms, nrVariables))
	#fomsPerYear = np.zeros((nrFoms, seq_len, nrVariables))

	#fomsPerCaseDict = OrderedDict()
	fomsPerCaseDict = {}
	fomsPerCase = np.zeros((nrCases, nrFoms, nrVariables))
	#fomsPerCase = np.zeros((nrFoms, nrCases, nrVariables))

	#print(np.where((targets == np.nan)))
	#print(np.where((predictions == np.nan)))

	fomsPerCaseMeanTbl = None
	
	# Compute the number of non-NaN rows also:
	nrNonNanRows = np.zeros((nrVariables, 1))

	for ii, thisVar in enumerate(targetVars):
		thisVarTargets = targets[:,:,ii]
		thisVarPreds = predictions[:,:,ii]
		
		# extract the index (= 0, 1 or 2) for the species of this targetVar:
		thisSpeciesIdx = [idx for idx, spc in enumerate(speciess) if spc in thisVar]
		
		# The DataFrames must be created here to assign the dictionary contents
		# properly (otherwise the dictionary points to the DataFrame object, and
		# the data fot the last target variable will be replicated into the
		# dictionary for all variables):
		fomsPerYearTbl = pd.DataFrame(columns=FOMs, index=range(1,seq_len+1))
		if siteID is not None:
			fomsPerCaseTbl = pd.DataFrame(columns=FOMs, index=siteID)
		else:
			fomsPerCaseTbl = pd.DataFrame(columns=FOMs, index=range(nrCases))
		
		# ====================================================================
		# Compute figures of merits per case, i.e. per field plot. These results
		# can be used to search best and worst time series predictions from the 
		# test set. Note: nbr of figures of merits nrFoms = 5.
		# --------------------------------------------------------------------
		fomDict, foms = compute_metrics(np.transpose(thisVarTargets), np.transpose(thisVarPreds))
		fomsThisVar = np.transpose(foms)
		yTrueMean = fomDict['y_true_mean']
		yTrueMean = np.expand_dims(yTrueMean, 1)
		fomsThisVar = np.concatenate((fomsThisVar, yTrueMean), axis=1)
		
		if exstFile is not None:
			# Assign NaN's to the cases for which this species did not exist on the stand:
			nullIdx = np.where(exstTbl[:,thisSpeciesIdx] == 0)
			fomsThisVar[nullIdx,:] = np.nan
			# The number of valid rows for this species data (not tested 19.3.2024):
			nrNonNanRows[ii] = exstTbl.shape[0] - nullIdx[0].shape[0]
		
		fomsPerCase[:,:,ii] = fomsThisVar
		fomsPerCaseTbl.loc[:,:] = fomsThisVar
		fomsPerCaseDict[thisVar] = fomsPerCaseTbl
		
		# Compute the mean over the all cases (skip NaN's) for this variable:
		fomsPerCaseMeanTbl = pd.concat([fomsPerCaseMeanTbl, fomsPerCaseTbl.mean(axis=0)], axis=1)
		## Compute the number of (non-NaN) cases also:
		#nrNonNanRows_cases = countNonNanRows(fomsPerCaseTbl)
		
		if outPath is not None:
			outFile = 'fomsPerCase_' + thisVar + '.csv'
			fomsPerCaseTbl.to_csv(os.path.join(outPath, outFile), sep = ',', index = True)
			
		# ====================================================================
		# If the presence/absence table was given, replace the missing species
		# data with NaN's:
		# --------------------------------------------------------------------
		if exstFile is not None:
			# Remove the rows of targets & predictions where this species did not exist:
			nullIdx = np.where(exstTbl[:,thisSpeciesIdx] == 0)
			thisVarTargets = np.delete(thisVarTargets, nullIdx, 0)
			thisVarPreds = np.delete(thisVarPreds, nullIdx, 0)
		
		# ====================================================================
		# Compute figures of merits per variable by flattening all data:
		# --------------------------------------------------------------------
		fomDict, foms = compute_metrics(thisVarTargets.flatten(), thisVarPreds.flatten())
		foms = np.concatenate((foms.flatten(), fomDict['y_true_mean']), axis=0)
		fomsFlat[ii,:] = foms
		#fomsFlat[ii,:] = foms.flatten()
		# compute_metrics should return a 5 x 1 (nrFoms x 1) array:
		#fomsFlatTbl[thisVar] = fomsFlat[:,ii]

		# ====================================================================
		# Compute figures of merits per variable per year, i.e. by interpreting
		# the yearly results (1 - 25) as separate 'sub-variables' of the same
		# physical variable (H_pine_1.0, H_pine_2.0, etc. are 
		# --------------------------------------------------------------------
		fomDict, foms = compute_metrics(thisVarTargets, thisVarPreds)
		yTrueMean = fomDict['y_true_mean']
		yTrueMean = np.expand_dims(yTrueMean, 0)
		foms = np.concatenate((foms, yTrueMean), axis=0)
		
		fomsPerYear[:,:,ii] = np.transpose(foms)
		fomsPerYearTbl.loc[:,:] = np.transpose(foms)
		fomsPerYearDict[thisVar] = fomsPerYearTbl
		if outPath is not None:
			outFile = 'fomsPerYear_' + thisVar + '.csv'
			fomsPerYearTbl.to_csv(os.path.join(outPath, outFile), sep = ',', index = True)

	fomsFlatTbl.loc[:,:] = fomsFlat

	# Finalize the mean per case table:
	# Note: The mean of the target values per each case is different, thus
	# computing the mean of absolute RMSE & BIAS values is problematic.
	fomsPerCaseMeanTbl = fomsPerCaseMeanTbl.T
	fomsPerCaseMeanTbl.index = targetVars
	# Remove the means of the absolute FoM's:
	fomsPerCaseMeanTbl = fomsPerCaseMeanTbl[['RMSEp', 'BIASp', 'R2', 'Ymean']]
	fomsPerCaseMeanTbl['N'] = nrNonNanRows
	fomsPerCaseDict['mean'] = fomsPerCaseMeanTbl
		
	if outPath is not None:
		fomsFlatTbl.to_csv(os.path.join(outPath, 'fomsFlat.csv'), sep = ',', index = True)
		fomsPerCaseMeanTbl.to_csv(os.path.join(outPath, 'fomsPerCaseMean.csv'), sep = ',', index = True)

	return fomsFlat, fomsPerYear, fomsPerCase, fomsFlatTbl, fomsPerYearDict, fomsPerCaseDict




def computeResults_old(targets, predictions, targetVars, FOMs = ['RMSE', 'RMSEp', 'BIAS', 'BIASp', 'R2', 'Ymean'], exstFile = None, outPath = None):

	nrCases = targets.shape[0]
	seq_len = targets.shape[1]
	nrVariables = targets.shape[2]
	nrFoms = len(FOMs)

	# Extract the species presence/absence table from the saved test data 
	# set file. The table contains indicator if the species at hand was
	# present (= 1) or absent (= 0) at the site (stand):
	if exstFile is not None:
		exstTbl = pd.read_csv(exstFile)
		# Get also the siteID from exstFile, if given:
		siteID = exstTbl['siteID'].values
		exstTbl = exstTbl[['exst_pine', 'exst_spr', 'exst_bl']].values
	else:
		siteID = None
		
	fomsFlatTbl = pd.DataFrame(columns=FOMs, index=targetVars)
	#fomsFlatTbl = pd.DataFrame(index=FOMs)
	fomsFlat = np.zeros((nrVariables, nrFoms))

	#fomsPerYearDict = OrderedDict()
	fomsPerYearDict = {}
	fomsPerYear = np.zeros((seq_len, nrFoms, nrVariables))
	#fomsPerYear = np.zeros((nrFoms, seq_len, nrVariables))

	#fomsPerCaseDict = OrderedDict()
	fomsPerCaseDict = {}
	fomsPerCase = np.zeros((nrCases, nrFoms, nrVariables))
	#fomsPerCase = np.zeros((nrFoms, nrCases, nrVariables))

	#print(np.where((targets == np.nan)))
	#print(np.where((predictions == np.nan)))
	
	fomsPerCaseMeanTbl = None

	for ii, thisVar in enumerate(targetVars):
		thisVarTargets = targets[:,:,ii]
		thisVarPreds = predictions[:,:,ii]
		
		# The DataFrames must be created here to assign the dictionary contents
		# properly (otherwise the dictionary points to the DataFrame object, and
		# the data fot the last target variable will be replicated into the
		# dictionary for all variables):
		fomsPerYearTbl = pd.DataFrame(columns=FOMs, index=range(1,seq_len+1))
		if siteID is not None:
			fomsPerCaseTbl = pd.DataFrame(columns=FOMs, index=siteID)
		else:
			fomsPerCaseTbl = pd.DataFrame(columns=FOMs, index=range(nrCases))
		
		# ====================================================================
		# Compute figures of merits per case, i.e. per field plot. These results
		# can be used to search best and worst time series predictions from the 
		# test set. Note: nbr of figures of merits nrFoms = 5.
		# --------------------------------------------------------------------
		fomDict, foms = compute_metrics(np.transpose(thisVarTargets), np.transpose(thisVarPreds))
		fomsThisVar = np.transpose(foms)
		yTrueMean = fomDict['y_true_mean']
		yTrueMean = np.expand_dims(yTrueMean, 1)
		fomsThisVar = np.concatenate((fomsThisVar, yTrueMean), axis=1)
		
		if exstFile is not None:
			# Assign NaN's to the cases for which this species did not exist on the stand:
			nullIdx = np.where(exstTbl[:,ii] == 0)
			fomsThisVar[nullIdx,:] = np.nan
		
		fomsPerCase[:,:,ii] = fomsThisVar
		fomsPerCaseTbl.loc[:,:] = fomsThisVar
		fomsPerCaseDict[thisVar] = fomsPerCaseTbl
		
		# Compute the mean over the all cases (skip NaN's) for this variable:
		fomsPerCaseMeanTbl = pd.concat([fomsPerCaseMeanTbl, fomsPerCaseTbl.mean(axis=0)], axis=1)
		
		if outPath is not None:
			outFile = 'fomsPerCase_' + thisVar + '.csv'
			fomsPerCaseTbl.to_csv(os.path.join(outPath, outFile), sep = ',', index = True)
			
		# ====================================================================
		# If the presence/absence table was given, replace the missing species
		# data with NaN's:
		# --------------------------------------------------------------------
		if exstFile is not None:
			# Remove the rows of targets & predictions where this species did not exist:
			nullIdx = np.where(exstTbl[:,ii] == 0)
			thisVarTargets = np.delete(thisVarTargets, nullIdx, 0)
			thisVarPreds = np.delete(thisVarPreds, nullIdx, 0)
		
		# ====================================================================
		# Compute figures of merits per variable by flattening all data:
		# --------------------------------------------------------------------
		fomDict, foms = compute_metrics(thisVarTargets.flatten(), thisVarPreds.flatten())
		fomsFlat[ii,:] = foms.flatten()
		# compute_metrics should return a 5 x 1 (nrFoms x 1) array:
		#fomsFlatTbl[thisVar] = fomsFlat[:,ii]

		# ====================================================================
		# Compute figures of merits per variable per year, i.e. by interpreting
		# the yearly results (1 - 25) as separate 'sub-variables' of the same
		# physical variable (H_pine_1.0, H_pine_2.0, etc. are 
		# --------------------------------------------------------------------
		foo, foms = compute_metrics(thisVarTargets, thisVarPreds)
		fomsPerYear[:,:,ii] = np.transpose(foms)
		fomsPerYearTbl.loc[:,:] = np.transpose(foms)
		fomsPerYearDict[thisVar] = fomsPerYearTbl
		if outPath is not None:
			outFile = 'fomsPerYear_' + thisVar + '.csv'
			fomsPerYearTbl.to_csv(os.path.join(outPath, outFile), sep = ',', index = True)

	fomsFlatTbl.loc[:,:] = fomsFlat

	# Finalize the mean per case table:
	# Note: The mean of the target values per each case is different, thus
	# computing the mean of absolute RMSE & BIAS values is problematic.
	fomsPerCaseMeanTbl = fomsPerCaseMeanTbl.T
	fomsPerCaseMeanTbl.index = targetVars
	# Remove the means of the absolute FoM's:
	fomsPerCaseMeanTbl = fomsPerCaseMeanTbl[['RMSEp', 'BIASp', 'R2']]
	fomsPerCaseDict['mean'] = fomsPerCaseMeanTbl
		
	if outPath is not None:
		fomsFlatTbl.to_csv(os.path.join(outPath, 'fomsFlat.csv'), sep = ',', index = True)
		fomsPerCaseMeanTbl.to_csv(os.path.join(outPath, 'fomsPerCaseMean.csv'), sep = ',', index = True)
	
	return fomsFlat, fomsPerYear, fomsPerCase, fomsFlatTbl, fomsPerYearDict, fomsPerCaseDict


	
def computeResults_oldest(targets, predictions, targetVars, FOMs = ['RMSE', 'RMSEp', 'BIAS', 'BIASp', 'R2'], exstFile = None, outPath = None):

	nrCases = targets.shape[0]
	seq_len = targets.shape[1]
	nrVariables = targets.shape[2]
	nrFoms = len(FOMs)

	# Extract the species presence/absence table from the saved test data 
	# set file. The table contains indicator if the species at hand was
	# present (= 1) or absent (= 0) at the site (stand):
	if exstFile is not None:
		exstTbl = pd.read_csv(exstFile)
		exstTbl = exstTbl[['exst_pine', 'exst_spr', 'exst_bl']].values
		# Replace the zeros with NaN's:
		exstTbl = np.where(exstTbl==0, np.nan, exstTbl)

	fomsFlatTbl = pd.DataFrame(columns=FOMs, index=targetVars)
	#fomsFlatTbl = pd.DataFrame(index=FOMs)
	fomsFlat = np.zeros((nrVariables, nrFoms))

	#fomsPerYearDict = OrderedDict()
	fomsPerYearDict = {}
	fomsPerYear = np.zeros((seq_len, nrFoms, nrVariables))
	#fomsPerYear = np.zeros((nrFoms, seq_len, nrVariables))

	#fomsPerCaseDict = OrderedDict()
	fomsPerCaseDict = {}
	fomsPerCase = np.zeros((nrCases, nrFoms, nrVariables))
	#fomsPerCase = np.zeros((nrFoms, nrCases, nrVariables))

	for ii, thisVar in enumerate(targetVars):
		# The DataFrames must be created here to assign the dictionary contents
		# properly (otherwise the dictionary points to the DataFrame object, and
		# the data fot the last target variable will be replicated into the
		# dictionary for all variables):
		fomsPerYearTbl = pd.DataFrame(columns=FOMs, index=range(1,seq_len+1))
		fomsPerCaseTbl = pd.DataFrame(columns=FOMs, index=range(nrCases))
		
		# ====================================================================
		# If the presence/absence table was given, replace the missing species
		# data with NaN's:
		# --------------------------------------------------------------------
		if exstFile is not None:
			# The NaN mask has to be replicated for seq_len first:
			thisVarMask = np.repeat(exstTbl[:,ii], seq_len, axis=1)
			targets[:,:,ii] = targets[:,:,ii] * thisVarMask
			predictions[:,:,ii] = predictions[:,:,ii] * thisVarMask
		
		# ====================================================================
		# Compute figures of merits per variable by flattening all data:
		# --------------------------------------------------------------------
		fomDict, foms = compute_metrics(targets[:,:,ii].flatten(), predictions[:,:,ii].flatten())
		fomsFlat[ii,:] = foms.flatten()
		# compute_metrics should return a 5 x 1 (nrFoms x 1) array:
		#fomsFlatTbl[thisVar] = fomsFlat[:,ii]

		# ====================================================================
		# Compute figures of merits per variable per year, i.e. by interpreting
		# the yearly results (1 - 25) as separate 'sub-variables' of the same
		# physical variable (H_pine_1.0, H_pine_2.0, etc. are 
		# --------------------------------------------------------------------
		foo, foms = compute_metrics(targets[:,:,ii], predictions[:,:,ii])
		fomsPerYear[:,:,ii] = np.transpose(foms)
		fomsPerYearTbl.loc[:,:] = np.transpose(foms)
		fomsPerYearDict[thisVar] = fomsPerYearTbl
		if outPath is not None:
			outFile = 'fomsPerYear_' + thisVar + '.csv'
			fomsPerYearTbl.to_csv(os.path.join(outPath, outFile), sep = ',', index = True)

		# ====================================================================
		# Compute figures of merits per case, i.e. per field plot. These results
		# can be used to search best and worst time series predictions from the 
		# test set. Note: nbr of figures of merits nrFoms = 5.
		# --------------------------------------------------------------------
		foo, foms = compute_metrics(np.transpose(targets[:,:,ii]), np.transpose(predictions[:,:,ii]))
		fomsPerCase[:,:,ii] = np.transpose(foms)
		fomsPerCaseTbl.loc[:,:] = np.transpose(foms)
		fomsPerCaseDict[thisVar] = fomsPerCaseTbl
		if outPath is not None:
			outFile = 'fomsPerCase_' + thisVar + '.csv'
			fomsPerCaseTbl.to_csv(os.path.join(outPath, outFile), sep = ',', index = False)

	fomsFlatTbl.loc[:,:] = fomsFlat
	if outPath is not None:
		fomsFlatTbl.to_csv(os.path.join(outPath, 'fomsFlat.csv'), sep = ',', index = True)
		
	return fomsFlat, fomsPerYear, fomsPerCase, fomsFlatTbl, fomsPerYearDict, fomsPerCaseDict	


# compute_metrics()
#
# This function computes the metrics RMSE, BIAS ,... for regression
# type predictions. 
#
# Inputs:
#
# y_true	(numpy array) A two dimensional Numpy array of true values with variables
#			as columns, and cases as rows.
#
# y_pred	(numpy array) A two dimensional Numpy array of predicted values with variables
#			as columns, and cases as rows. (size identical with y-true)
#
# Output:
#
# fomDict	(dict) A dictionary with the metrics: rmse, rmse%, bias, bias% and R2.
#			Also the mean values of these (computed over the different variables) 
#			will be returned.
#
# fomArr	(Numpy array) An alternative output format. The figures of merit will
#			be returned all in a 2D Numpy array of size [nrFoms x nrVariables].
#			nrFoms = 5, with present performace figures (= RMSE, RMSE%, BIAS, BIAS%, R2)

def compute_metrics(y_true, y_pred):

	# Ensure that the input arrays are two-dimensional:
	if y_true.ndim == 1:
		y_true = np.expand_dims(y_true, 1)
	if y_pred.ndim == 1:
		y_pred = np.expand_dims(y_pred, 1)

	num_variables = y_true.shape[1]
	# Define lists for output dictionary:
	rmse_list, bias_list, r2_list, rmsep_list, biasp_list = [], [], [], [], []
	# Define Numpy arrays for alternative outputs (RMSE, RMSE%, BIAS, BIAS%, R2):
	nrFoms = 5
	fomArr = np.zeros((nrFoms,num_variables))

	y_true_mean = np.mean(y_true, axis=0)
	# Assign y_true_mean zero elements to -1, if any:
	#y_true_mean[y_true_mean < 1e-6] = -1

	for i in range(num_variables):
		y_true_variable = y_true[:, i]
		y_pred_variable = y_pred[:, i]

		# Compute RMSE
		rmse = np.sqrt(mean_squared_error(y_true_variable, y_pred_variable))
		rmse_list.append(rmse)
		fomArr[0,i] = rmse

		# Compute Bias
		bias = np.mean(y_pred_variable - y_true_variable)
		bias_list.append(bias)
		fomArr[2,i] = bias
		
		# Assure that y_true_mean[i] is positive to prevent
		# change of sign for relative errors:
		if y_true_mean[i] < 0:
			y_true_mean[i] *= -1

		# Compute rmse% & bias% + check for zero division:
		if y_true_mean[i] == 0:
			rmsep = -99.0
			biasp = -99.0
		else:
			rmsep = 100*rmse/y_true_mean[i]
			biasp = 100*bias/y_true_mean[i]
		rmsep_list.append(rmsep)
		biasp_list.append(biasp)
		fomArr[1,i] = rmsep
		fomArr[3,i] = biasp

		# Compute r2-score
		r2 = r2_score(y_true_variable, y_pred_variable)
		r2 = max(r2, 0)
		r2_list.append(r2)
		fomArr[4,i] = r2

	N = y_true.shape[0]
	rmse_mean = sum(rmse_list)/len(rmse_list)
	rmsep_mean = sum(rmsep_list)/len(rmsep_list)
	bias_mean = sum(bias_list)/len(bias_list)
	biasp_mean = sum(biasp_list)/len(biasp_list)
	r2_mean = sum(r2_list)/len(r2_list)

	fomDict = {
		'RMSE': rmse_list,
		'RMSEp': rmsep_list,
		'BIAS': bias_list,
		'BIASp': biasp_list,
		'R2': r2_list,
		'RMSE_mean': rmse_mean,
		'RMSEp_mean': rmsep_mean,
		'BIAS_mean': bias_mean,
		'BIASp_mean': biasp_mean,
		'R2_mean': r2_mean,
		'N': N,
		'y_true_mean': y_true_mean
		}

	return fomDict, fomArr
	
	

def countNonNanRows(df):
	# Create a boolean mask 'nonNanRow' where each True value 
	# represents a non-NaN value in the row
	nonNanRow = df.notna().all(axis=1)

	# Count the non-NaN rows using the boolean mask:
	count = nonNanRow.sum()

	return count


	
def compute_and_save_normalization_params(data, save_path):
    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Compute mean and variance for normalization
    scaler.fit(data)

    # Save normalization parameters to a text file
    np.savetxt(save_path, np.vstack((scaler.mean_, scaler.var_)), delimiter=',')

'''
# Example usage
# Replace this data with your actual data
data_to_normalize = np.random.rand(100, 3)  # 100 samples, 3 features

# Specify the path to save normalization parameters
save_path = 'normalization_params.txt'

# Compute and save normalization parameters
compute_and_save_normalization_params(data_to_normalize, save_path)
'''



from sklearn.preprocessing import StandardScaler

def read_and_inverse_transform(data, normalization_params_path):
    # Load normalization parameters from the text file
    mean, var = np.loadtxt(normalization_params_path, delimiter=',')

    # Create a StandardScaler instance with loaded parameters
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.var_ = var

    # Perform inverse transform
    inverse_transformed_data = scaler.inverse_transform(data)

    return inverse_transformed_data

'''
# Example usage
# Replace this data with your actual data
data_to_inverse_transform = np.random.rand(10, 3)  # 10 samples, 3 features

# Specify the path to read normalization parameters
read_path = 'normalization_params.txt'

# Perform inverse transform using loaded normalization parameters
inverse_transformed_data = read_and_inverse_transform(data_to_inverse_transform, read_path)

# Print the result
print("Original Data:\n", data_to_inverse_transform)
print("\nInverse Transformed Data:\n", inverse_transformed_data)
'''



# targetDataNorm()
#
# This fumction computes the target data normalization coefficients
# (mode = 'fit') and computes the normalized target data using the
# computed StandardScaler() coefficients. Both operations may be
# performed for training data (only) on a single run.
#
# Inputs:
#
# targetData	(2D pandas dataFrame) The dataframe containing the
#				target data. As the data for each variable occupies
#				'nYears' columns, the data has to be organized first
#				to one column per variable for computing the coefficients
#				(mode = 'fit'). 
#
#				When normalizing the data the data may be
#				in original 2D shape for each variable, but the
#				provided StandardScaler() coefficients have to be
# 				extracted variable-by-variable for normalization.
#
#				Note 1. that the StandardScaler() does not include NaNs
#				in the computations (which is pretty obvious).
#
#				Note 2. Changing NaN values to non-NaN values (default = 
#				zeros) takes place before data normalization, but the
#				computation of coefficients has to be done without
#				NaNs, as explained above.
#
#				Note 3: The input target data is in Pandas DataFrame
#				format to select the correct variable columns by name.
#				However, the normalized output is a Numpy 2D array.
#
# targetVars	(list of strings) The target variable names in the target
#				data.
#				Note that each target variable occupies 'nYears' columns
#				in the data.
#
# statsFile		(file path) Path (= folder + filename) to normalization 
#				statistics file.
#
# nYears		(int) Number of prediction years (default = 25).
#
# mode			(string) The mode of operation:
#
#				'fit': Compute normalization coefficients for the input data.
#				'transform': Normalize the input data using the provided
#							 coefficiens.
#				'fitEtTxform': Compute the coefficients and perform normalizing
#							on a single run (for training data only).
#				'invTransform': De-normalize the input data using the provided
#							 coefficiens. (to be issued)
#
# Outputs:
#
# normData		(Numpy 2D array) If mode = 'fit', then normData = None.
#				If If mode = 'transform', then normData contains the normalized 
#				target data.

def targetDataNorm(targetData, targetVars, statsFile = None, nYears = 25, mode = 'fit', replaceNans = 0.0):

	assert statsFile is not None, \
		"targetDataNorm: Statistics file name not specified. Aborting!"
	
	tgtDataScaler = StandardScaler()
		
	targetVarCols = targetData.columns

	if mode == 'fit' or mode == 'fitEtTxform':
		tgtVarData4Norm = np.zeros((targetData.shape[0]*nYears, len(targetVars)))

		# Flatten all nYears columns of target data (per tgtVar) to compute the statistics: 
		for ii, thisTgtVar in enumerate(targetVars):
			thisTgtVarCols = [hdr for hdr in targetVarCols if thisTgtVar in hdr]
			thisTgtVarData = targetData[thisTgtVarCols].values.flatten()
			tgtVarData4Norm[:,ii] = thisTgtVarData

		tgtDataScaler.fit(tgtVarData4Norm)
		print(tgtDataScaler.mean_)
		print(tgtDataScaler.var_)
		
		# Save the normalization statistics to file:
		np.savetxt(statsFile, np.vstack((tgtDataScaler.mean_, tgtDataScaler.var_, tgtDataScaler.scale_)), delimiter=',')
		
		normData = None

	if mode == 'transform':
		# Read the normalization statistics from file:
		tgtDataStats = np.loadtxt(statsFile, delimiter= ',')
		tgtDataScaler.mean_ = tgtDataStats[0,:]
		tgtDataScaler.var_ = tgtDataStats[1,:]
		tgtDataScaler.scale_ = tgtDataStats[2,:]

	if mode == 'transform' or mode == 'fitEtTxform':
		normData = np.zeros(targetData.shape)
		print(normData.shape)
		txScaler = StandardScaler()

		if replaceNans is not None:
			# Replace the target data NaN values with provided number (default = 0.0):
			targetData = targetData.fillna(value = replaceNans)
		
		for ii, thisTgtVar in enumerate(targetVars):
			thisTgtVarCols = [hdr for hdr in targetVarCols if thisTgtVar in hdr]
			thisTgtVarColIdx = [idx for idx, hdr in enumerate(targetVarCols) if thisTgtVar in hdr]
			thisTgtVarData = targetData[thisTgtVarCols].values
			
			txScaler.mean_ = tgtDataScaler.mean_[ii]
			txScaler.var_ = tgtDataScaler.var_[ii]
			txScaler.scale_ = tgtDataScaler.scale_[ii]
			
			thisTgtVarDataNorm = txScaler.transform(thisTgtVarData)

			normData[:,thisTgtVarColIdx] = thisTgtVarDataNorm
			
	return normData



def targetDataNorm_old(targetData, targetVars, tgtDataScaler = None, nYears = 25, mode = 'fit', fileOut = None, replaceNans = 0.0):

	if tgtDataScaler == None:
		tgtDataScaler = StandardScaler()
		
	targetVarCols = targetData.columns

	if mode == 'fit' or mode == 'fitEtTxform':
		tgtVarData4Norm = np.zeros((targetData.shape[0]*nYears, len(targetVars)))

		for ii, thisTgtVar in enumerate(targetVars):
			thisTgtVarCols = [hdr for hdr in targetVarCols if thisTgtVar in hdr]
			thisTgtVarData = targetData[thisTgtVarCols].values.flatten()
			tgtVarData4Norm[:,ii] = thisTgtVarData

		tgtDataScaler.fit(tgtVarData4Norm)
		print(tgtDataScaler.mean_)
		print(tgtDataScaler.var_)
		
		normData = None

	if mode == 'transform' or mode == 'fitEtTxform':
		normData = np.zeros(targetData.shape)
		print(normData.shape)
		txScaler = StandardScaler()

		if replaceNans is not None:
			# Replace the target data NaN values with provided number (default = 0.0):
			targetData = targetData.fillna(value = replaceNans)
		
		for ii, thisTgtVar in enumerate(targetVars):
			thisTgtVarCols = [hdr for hdr in targetVarCols if thisTgtVar in hdr]
			thisTgtVarColIdx = [idx for idx, hdr in enumerate(targetVarCols) if thisTgtVar in hdr]
			thisTgtVarData = targetData[thisTgtVarCols].values
			
			txScaler.mean_ = tgtDataScaler.mean_[ii]
			txScaler.var_ = tgtDataScaler.var_[ii]
			txScaler.scale_ = tgtDataScaler.scale_[ii]
			
			thisTgtVarDataNorm = txScaler.transform(thisTgtVarData)

			normData[:,thisTgtVarColIdx] = thisTgtVarDataNorm
			
	if fileOut is not None:
		np.savetxt(fileOut, np.vstack((tgtDataScaler.mean_, tgtDataScaler.var_, tgtDataScaler.scale_)), delimiter=',')

	return tgtDataScaler, normData



# Ne-normalize the targets and teh prediction outputs.
#
# targets		(3D numpy array)
# preds			(3D numpy array)
# statsFile 	(File path)

def predsdeNorm(targets, preds, statsFile = None):

	assert statsFile is not None, \
		"predsdeNorm: Statistics file name not specified. Aborting!"
		
	targets_denorm = np.zeros(targets.shape)
	preds_denorm = np.zeros(preds.shape)
	
	txScaler = StandardScaler()
	tgtDataScaler = StandardScaler()
	
	# Read the normalization statistics from file:
	tgtDataStats = np.loadtxt(statsFile, delimiter= ',')
	tgtDataScaler.mean_ = tgtDataStats[0,:]
	tgtDataScaler.var_ = tgtDataStats[1,:]
	tgtDataScaler.scale_ = tgtDataStats[2,:]

	for ii in range(targets.shape[2]):
	
		txScaler.mean_ = tgtDataScaler.mean_[ii]
		txScaler.var_ = tgtDataScaler.var_[ii]
		txScaler.scale_ = tgtDataScaler.scale_[ii]
		
		targets_denorm[:,:,ii] = txScaler.inverse_transform(targets[:,:,ii])
		preds_denorm[:,:,ii] = txScaler.inverse_transform(preds[:,:,ii])

	return targets_denorm, preds_denorm


def predsdeNorm_old(targets, preds, tgtDataScaler):

	targets_denorm = np.zeros(targets.shape)
	preds_denorm = np.zeros(preds.shape)
	
	txScaler = StandardScaler()

	for ii in range(targets.shape[2]):
	
		txScaler.mean_ = tgtDataScaler.mean_[ii]
		txScaler.var_ = tgtDataScaler.var_[ii]
		txScaler.scale_ = tgtDataScaler.scale_[ii]
		
		targets_denorm[:,:,ii] = txScaler.inverse_transform(targets[:,:,ii])
		preds_denorm[:,:,ii] = txScaler.inverse_transform(preds[:,:,ii])

	return targets_denorm, preds_denorm



def targetDataNorm_dev(tgtDataScaler, targetData, targetVars, nYears = 25, mode = 'fit'):

	#tgtDataScaler = StandardScaler()

	if mode == 'fit' or mode == 'fitEtTxform':
		tgtVarData4Norm = np.zeros((targetData.shape[0]*nYears, len(targetVars)))

		for ii, thisTgtVar in enumerate(targetVars):
			thisTgtVarCols = [hdr for hdr in targetVarCols if thisTgtVar in hdr]
			thisTgtVarData = targetData[thisTgtVarCols].values.flatten()
			tgtVarData4Norm[:,ii] = thisTgtVarData

		tgtDataScaler.fit(tgtVarData4Norm)
		print(tgtDataScaler.mean_)
		print(tgtDataScaler.var_)
		
		normData = None

	if mode == 'transform' or mode == 'fitEtTxform':
		normData = targetData
		txScaler = StandardScaler()
		
		for ii, thisTgtVar in enumerate(targetVars):
			thisTgtVarCols = [hdr for hdr in targetVarCols if thisTgtVar in hdr]
			thisTgtVarData = targetData[thisTgtVarCols].values

			#print(thisTgtVarData[1:3,1:5])
			#print("")
			
			txScaler.mean_ = tgtDataScaler.mean_[ii]
			txScaler.var_ = tgtDataScaler.var_[ii]
			txScaler.scale_ = tgtDataScaler.scale_[ii]

			thisTgtVarDataNorm = txScaler.transform(thisTgtVarData)
			
			normData.loc[thisTgtVarCols,:] = thisTgtVarDataNorm
			#print(thisTgtVarDataNorm[1:3,1:5])
			#print("")

	return tgtDataScaler, normData


# joinTestDataWithResults()
#
# targets		(Numpy matrix of size [nrCases, seq_len, numVars])
# predictions	(Numpy matrix of size [nrCases, seq_len, numVars])
# inputDataFile	(path) Path to the input data file
# paramDict		(dict) The diftionary containing the test parameters.
#				must contain the items 'targetVars' and 'nYears'.

def joinTestDataWithResults(paramDict, inputDataFile, predictions, fomsPerCaseDict = None, targets = None, outFile = None):

	# Init random generator (for monitoring data coherence):
	rng = np.random.default_rng()
	nrMonitorCases = 10
	maxDiff = 5E-4

	#testDataFile = os.path.join(outPath, 'testSetData.csv')
	joinedDataOut = pd.read_csv(inputDataFile)
	#print("testData.shape = ", testData.shape)

	fomStrs = ['RMSE', 'RMSEp', 'BIAS', 'BIASp', 'R2', 'Ymean']

	for ii, thisTgtVar in enumerate (paramDict['targetVars']):
		thisTgtVarPredCols = [(thisTgtVar + '_' + 'p_' + str(x+1) + '.0') for x in range(paramDict['nYears'])]
		thisTgtVarPred_df = pd.DataFrame(data=predictions[:,:,ii], columns=thisTgtVarPredCols)
		joinedDataOut = pd.concat([joinedDataOut, thisTgtVarPred_df], axis=1)

	# Add the figures of merits per target variable, if desired:
	if fomsPerCaseDict is not None:
		for ii, thisTgtVar in enumerate (paramDict['targetVars']):
			fomStrsThisVar = [hdr + '_' + thisTgtVar for hdr in fomStrs]
			fomsPerCaseThisVar_df = pd.DataFrame(data= fomsPerCaseDict[thisTgtVar].values, columns=fomStrsThisVar)
			joinedDataOut = pd.concat([joinedDataOut, fomsPerCaseThisVar_df], axis=1)

	# And the original target values for monotoring, if desired:
	if targets is not None:
		for ii, thisTgtVar in enumerate (paramDict['targetVars']):
			thisTgtVarTgtCols = [(thisTgtVar + '_' + 't_' + str(x+1) + '.0') for x in range(paramDict['nYears'])]
			thisTgtVarTgt_df = pd.DataFrame(data=targets[:,:,ii], columns=thisTgtVarTgtCols)
			joinedDataOut = pd.concat([joinedDataOut, thisTgtVarTgt_df], axis=1)
			#joinedDataOut[thisTgtVarTgtCols] = targets[:,:,ii]

		# Monitor the coherence by comparing the target variables just added 
		# and the targets originally in inputDataFile (shall be identical):
		
		# 'Original' target data colunm indices:
		origTgtVarCols = [(thisTgtVar + '_' + str(x+1) + '.0') for x in range(paramDict['nYears'])]
		origTgtVarColIdx = [idx for idx, hdr in enumerate(joinedDataOut.columns) if hdr in origTgtVarCols]
		
		# ... and the just added target variable data column indices:
		tgtVarColIdx = [idx for idx, hdr in enumerate(joinedDataOut.columns) if hdr in thisTgtVarTgtCols]

		# Produce random indices to check the data coherence:
		idxs = np.arange(0, joinedDataOut.shape[0])
		rng.shuffle(idxs)
		# Take a subset of size 'nrClimIDs' of the shuffled starting years:
		idxs = idxs[0:nrMonitorCases]
		for jj, idx in enumerate(idxs):
			# Compute the sum of differences for the selected case's target variable components:
			diffTargets = np.sum(joinedDataOut.iloc[idx,origTgtVarColIdx].values - joinedDataOut.iloc[idx,tgtVarColIdx].values)
			if diffTargets > maxDiff:
				print("Target variable data not coherent!", diffTargets)
			else:
				print("Target variable coherence OK!", diffTargets)

	# Save the joined data set:
	if outFile is None:
		#outPath = os.path.dirname(inputDataFile)
		outPath, filename = os.path.split(inputDataFile)
		ff, ext = os.path.splitext(filename)
		outFile = os.path.join(outPath, ff + '_wPredictions.csv')
		
	joinedDataOut.to_csv(outFile, index=False)

	return joinedDataOut



def computeResults_ori(targets, predictions, targetVars, FOMs = ['RMSE', 'RMSEp', 'BIAS', 'BIASp', 'R2']):

	nrCases = targets.shape[0]
	seq_len = targets.shape[1]
	nrVariables = targets.shape[2]
	nrFoms = len(FOMs)

	# ====================================================================
	# Compute figures of merits per variable by flattening all data:
	# --------------------------------------------------------------------
	fomsAllTbl = pd.DataFrame(index=FOMs)
	fomsAll = np.zeros((nrFoms, nrVariables))
	for ii, thisVar in enumerate(targetVars):
		fomDict, foms = compute_metrics(targets[:,:,ii].flatten(), predictions[:,:,ii].flatten())
		fomsAll[:,ii] = foms.flatten()
		# compute_metrics should return a 5 x 1 (nrFoms x 1) array:
		fomsAllTbl[thisVar] = fomsAll[:,ii]

	# ====================================================================
	# Compute figures of merits per variable per year, i.e. by interpreting
	# the yearly results (1 - 25) as separate 'sub-variables' of the same
	# physical variable (H_pine_1.0, H_pine_2.0, etc. are 
	# --------------------------------------------------------------------
	fomsPerYearTbl = pd.DataFrame(index=FOMs)
	fomsPerYear = np.zeros((nrFoms, seq_len, nrVariables))
	for ii in range(nrVariables):
		foo, fomsPerYear[:,:,ii] = compute_metrics(targets[:,:,ii], predictions[:,:,ii])

	# ====================================================================
	# Compute figures of merits per case, i.e. per field plot. These results
	# can be used to search best and worst time series predictions from the 
	# test set. Note: nbr of figures of merits nrFoms = 5.
	# --------------------------------------------------------------------
	fomsPerCaseTbl = pd.DataFrame(index=FOMs)
	fomsPerCase = np.zeros((nrFoms, nrCases, nrVariables))
	for ii in range(nrVariables):
		foo, fomsPerCase[:,:,ii] = compute_metrics(np.transpose(targets[:,:,ii]), np.transpose(predictions[:,:,ii]))

	return fomsAllTbl, fomsAll, fomsPerYear, fomsPerCase


def plotResults_Nx2(targetVar, targets, preds, metadata = None, xRanges = None, title = None, figsize = (8,6), margin_p = 0.05, outFile = None, printFoMs = False, fontSizDict = None):

	commonCols = ['siteID', 'climID_orig', 'scenario', 'year_start', 'year_end', 'age_pine', 'age_spr', 'age_bl', 'H_pine', 'H_spr', 'H_bl', 'D_pine', 'D_spr', 'D_bl', 'BA_pine', 'BA_spr', 'BA_bl', 'siteType']

	nrPlots = targets.shape[0]
	if nrPlots > 1:
		fig, axs = plt.subplots(math.ceil(nrPlots/2), 2, figsize = figsize, layout='constrained')
	else:
		fig, axs = plt.subplots(1, 1, figsize = figsize, layout='constrained')

	# Extract species string; this assumes that species id string is followed by
	# the underscore in the variable's name 
	speciesStr = targetVar[targetVar.index('_')+1:]

	# Compose the forest variable column names (for the current species):
	forVarHdrs = ['age', 'H', 'D', 'BA']
	forVarHdrs = [hdr + speciesStr for hdr in forVarHdrs]
	# ... and the corresponding dataframe column indices:
	
	# -----------------------------------------------------
	# y-axis units:
	# -----------------------------------------------------
	tgtVarUnits = ''
	if 'D' in targetVar:
		tgtVarUnits = '[$cm$]'
	if 'H' in targetVar:
		tgtVarUnits = '[$m$]'
	if 'BA' in targetVar:
		tgtVarUnits = '[$m^2/ha$]'
	if 'V' in targetVar:
		tgtVarUnits = '[$m^3/ha$]'
	if 'npp' or 'NEP'or 'GPP' in targetVar:
		tgtVarUnits = '[$gC/(m^2 * y)$]'
	if 'GGrowth' in targetVar:
		tgtVarUnits = '[$m^3/(ha * y)$]'
	# -----------------------------------------------------
	
	if fontSizDict is not None:
		fontSiz_xLabel = fontSizDict['fontSiz_xLabel']
		fontSiz_yLabel = fontSizDict['fontSiz_yLabel']
		fontSiz_xTicks = fontSizDict['fontSiz_xTicks']
		fontSiz_yTicks = fontSizDict['fontSiz_yTicks']
		fontSiz_legend = fontSizDict['fontSiz_legend']
	else:
		fontSiz_xLabel = 16
		fontSiz_yLabel = 16
		fontSiz_xTicks = 14
		fontSiz_yTicks = 14
		fontSiz_legend = 14

	for ctr in range(nrPlots):
		if nrPlots > 2:
			ax = axs[int(ctr/2), int(ctr%2)]
		else:
			if nrPlots == 1:
				ax = axs
			else:
				ax = axs[ctr]
		
		ax.ticklabel_format(useOffset=False)
		
		tgtTimeSeries = targets[ctr,:]
		predTimeSeries = preds[ctr,:]
		N = targets.shape[1]

		#xRange = range(xRanges[ctr,0],xRanges[ctr,1]+1)
		
		#if xRanges is not None:
		#    xRange = range(xRanges[ctr,0],xRanges[ctr,1]+1)
		#else:
		#    xRange = range(1,len(tgtTimeSeries)+1)
		
		axxMin_x = xRanges[ctr,0]
		axxMax_x = xRanges[ctr,1]+1
		
		ax.plot(range(1,len(tgtTimeSeries)+1), tgtTimeSeries, label='Target (Prebasso)')
		ax.plot(range(1,len(tgtTimeSeries)+1), predTimeSeries, label='Prediction (FC_RNN)')

		xLabels = []
		for ii, jj in enumerate(range(axxMin_x, axxMax_x+1,5)):
			xLabels.append(str(jj))
		#print(xLabels)

		# Set the actual year labels (xticks). Note: extend the 
		ax.set_xticks(range(1,len(tgtTimeSeries)+2,5), labels=xLabels)
		plt.yticks(fontsize=fontSiz_yTicks)
		plt.xticks(fontsize=fontSiz_xTicks)
	   
		maxVal = max(max(tgtTimeSeries), max(predTimeSeries))
		minVal = min(min(tgtTimeSeries), min(predTimeSeries))
		valRange = maxVal - minVal
		axxMin_y = minVal-margin_p*valRange
		axxMax_y = maxVal+margin_p*valRange

		if np.isinf(axxMin_y) or np.isinf(axxMax_y) or np.isnan(axxMin_y) or np.isnan(axxMax_y):
			continue
		else:
			ax.set_ylim(axxMin_y, axxMax_y) # Plot with margin in y-direction

		ax.legend(fontsize = fontSiz_legend)
		ax.set_xlabel('Year', fontsize=fontSiz_xLabel)
		
		ylabelStr = targetVar.replace('_', ' / ')
		ylabelStr = ylabelStr.replace('spr', 'spruce')
		ylabelStr = ylabelStr.replace('bl', 'broadleaved')
		ylabelStr = ylabelStr + ' ' + tgtVarUnits
		ax.set_ylabel(ylabelStr, fontsize=fontSiz_yLabel)
		
		if printFoMs:
			fomDict, fomArr = compute_metrics(tgtTimeSeries, predTimeSeries)
			ax.text(1, maxVal-0.2*valRange, "RMSE%: {0:.1f}".format(fomDict['RMSEp'][0]), fontdict=None)
			ax.text(1, maxVal-0.25*valRange, "BIAS%: {0:.1f}".format(fomDict['BIASp'][0]), fontdict=None)
			ax.text(1, maxVal-0.3*valRange, "R2: {0:.2f}".format(fomDict['R2'][0]), fontdict=None)
			ax.text(1, maxVal-0.35*valRange, "N: {0:.0f}".format(N), fontdict=None)
		
		ax.tick_params(axis='x', length=5, direction='in', width=1)
		ax.tick_params(axis='y', length=5, direction='in', width=1)
		#ax.tick_params(axis='x', color='m', length=4, direction='in', width=1,
        #              labelcolor='g', grid_color='b')

		if metadata is not None:
			climId_idx = commonCols.index('climID_orig')
			scenario_idx = commonCols.index('scenario')
			siteType_idx = commonCols.index('siteType')
			age_idx = commonCols.index('age_'+speciesStr)
			H_idx = commonCols.index('H_'+speciesStr)
			D_idx = commonCols.index('D_'+speciesStr)
			BA_idx = commonCols.index('BA_'+speciesStr)
			
			xStartPos = len(tgtTimeSeries)-0.32*len(tgtTimeSeries)
			#ax.text(xStartPos, axxMin_y+0.4*valRange, "climID:  {0:.0f}".format(metadata.iloc[ctr,climId_idx]), fontdict=None)
			ax.text(xStartPos, axxMin_y+0.45*valRange, "Model: hadgem3_gc31_ll", fontdict=None)
			ax.text(xStartPos, axxMin_y+0.4*valRange, "Scenario: {scenario}".format(scenario=metadata.iloc[ctr,scenario_idx]), fontdict=None)
			
			ax.text(xStartPos, axxMin_y+0.3*valRange, "Site status @Year: {0:.0f}:".format(axxMin_x-1), fontdict=None)
			ax.text(xStartPos, axxMin_y+0.25*valRange, "Age: {0:.0f} $yrs$".format(metadata.iloc[ctr,age_idx]), fontdict=None)
			ax.text(xStartPos, axxMin_y+0.2*valRange, "DBH: {0:.0f} $cm$".format(metadata.iloc[ctr,D_idx]), fontdict=None)
			ax.text(xStartPos, axxMin_y+0.15*valRange, "Height: {0:.0f} $m$".format(metadata.iloc[ctr,H_idx]), fontdict=None)
			ax.text(xStartPos, axxMin_y+0.1*valRange, "Basal area: {0:.0f} $m^2/ha$".format(metadata.iloc[ctr,BA_idx]), fontdict=None)
			ax.text(xStartPos, axxMin_y+0.05*valRange, "Fertility class: {0:.0f}".format(metadata.iloc[ctr,siteType_idx]), fontdict=None)
			#if 'H' in targetVar or 'D' in targetVar or 'BA' in targetVar:
			#	targetVarIdx = commonCols.index(targetVar)
			#	ax.text(xStartPos, axxMin_y+0.05*valRange, targetVar + ": {0:.0f}".format(metadata.iloc[ctr,targetVarIdx]), fontdict=None)
			
	if title is not None:
		fig.suptitle(title, fontsize=14)
		
	if outFile is not None:
		plt.savefig(outFile, dpi=600, format = 'png')
		#plt.savefig(outFile, transparent=True)

	plt.show()

	return None


	
def plotResults_2x2(targets, preds, targetVar, figSize = (8,6), margin_p = 0.05):

	fig, axs = plt.subplots(2, 2, figSize = figSize)

	for xx in range(2):
		for yy in range(2):

			tgtTimeSeries = targets[yy+2*xx,:]
			predTimeSeries = preds[yy+2*xx,:]
			N = targets.shape[1]

			axs[xx, yy].plot(range(1,len(tgtTimeSeries)+1), tgtTimeSeries, label='Target')
			axs[xx, yy].plot(range(1,len(tgtTimeSeries)+1), predTimeSeries, label='Prediction')
			
			maxVal = max(max(tgtTimeSeries), max(predTimeSeries))
			minVal = min(min(tgtTimeSeries), min(predTimeSeries))
			valRange = maxVal - minVal
			axxMin_y = minVal-margin_p*valRange
			axxMax_y = maxVal+margin_p*valRange

			axs[xx, yy].ylim(axxMin_y, axxMax_y) # Plot with margin in y-direction

			axs[xx, yy].legend()
			axs[xx, yy].xlabel('Year', fontsize=16)
			axs[xx, yy].ylabel(targetVar, fontsize=16)
			
			fomDict, fomArr = compute_metrics(tgtTimeSeries, predTimeSeries)
			axs[xx, yy].text(1, maxVal-0.2*valRange, "RMSE%: {0:.1f}".format(fomDict['RMSEp'][0]), fontdict=None)
			axs[xx, yy].text(1, maxVal-0.25*valRange, "BIAS%: {0:.1f}".format(fomDict['BIASp'][0]), fontdict=None)
			axs[xx, yy].text(1, maxVal-0.3*valRange, "R2: {0:.2f}".format(fomDict['R2'][0]), fontdict=None)
			axs[xx, yy].text(1, maxVal-0.35*valRange, "N: {0:.0f}".format(N), fontdict=None)

			axs[xx, yy].show()

	return None


def plotTimeSeriesResult(targets, preds, targetVars, figSize = (8,6), caseIdx = 0, variable = 0, margin_p = 0.05):

	fig = plt.figure(figsize=figSize, clear=True)

	tgtTimeSeries = targets[caseIdx,:, variable]
	predTimeSeries = preds[caseIdx,:, variable]
	N = targets.shape[1]

	plt.plot(range(1,len(tgtTimeSeries)+1), tgtTimeSeries, label='Target')
	plt.plot(range(1,len(tgtTimeSeries)+1), predTimeSeries, label='Prediction')
	
	maxVal = max(max(tgtTimeSeries), max(predTimeSeries))
	minVal = min(min(tgtTimeSeries), min(predTimeSeries))
	valRange = maxVal - minVal
	axxMin_y = minVal-margin_p*valRange
	axxMax_y = maxVal+margin_p*valRange

	plt.ylim(axxMin_y, axxMax_y) # Plot with margin in y-direction

	plt.legend()
	plt.xlabel('Year', fontsize=16)
	plt.ylabel(targetVars[variable], fontsize=16)
	
	fomDict, fomArr = compute_metrics(tgtTimeSeries, predTimeSeries)
	plt.text(1, maxVal-0.2*valRange, "RMSE%: {0:.1f}".format(fomDict['RMSEp'][0]), fontdict=None)
	plt.text(1, maxVal-0.25*valRange, "BIAS%: {0:.1f}".format(fomDict['BIASp'][0]), fontdict=None)
	plt.text(1, maxVal-0.3*valRange, "R2: {0:.2f}".format(fomDict['R2'][0]), fontdict=None)
	plt.text(1, maxVal-0.35*valRange, "N: {0:.0f}".format(N), fontdict=None)

	plt.show()

	return None


def plotTimeSeriesResults(targets, preds, targetVars, figSize = (8,6), startIdx = 0, variable = 0, maxIter = 1000, margin_p = 0.05):

	fig = plt.figure(figsize=figSize, clear=True)
	skipPlot = False

	for ii in range(maxIter):
		idx = startIdx + ii

		if not skipPlot:
			plotTimeSeriesResult(targets, preds, targetVars, figSize = figSize, caseIdx = idx, variable = variable, margin_p = margin_p)
		
		# tgtTimeSeries = targets[idx,:, variable]
		# predTimeSeries = preds[idx,:, variable]

		# plt.plot(range(1,len(tgtTimeSeries)+1), tgtTimeSeries, label='Target')
		# plt.plot(range(1,len(tgtTimeSeries)+1), predTimeSeries, label='Prediction')
		
		# maxVal = max(max(tgtTimeSeries), max(predTimeSeries))
		# minVal = min(min(tgtTimeSeries), min(predTimeSeries))
		# valRange = maxVal - minVal
		# axxMin_y = minVal-margin_p*valRange
		# axxMax_y = maxVal+margin_p*valRange

		# plt.ylim(axxMin_y, axxMax_y) # Plot with margin in y-direction

		# plt.legend()
		# plt.xlabel('Year', fontsize=16)
		# plt.ylabel(targetVars[variable], fontsize=16)
		
		# fomDict, fomArr = compute_metrics(tgtTimeSeries, predTimeSeries)
		# plt.text(1, maxVal-0.2*valRange, "RMSE%: {0:.1f}".format(fomDict['RMSEp'][0]), fontdict=None)
		# plt.text(1, maxVal-0.25*valRange, "BIAS%: {0:.1f}".format(fomDict['BIASp'][0]), fontdict=None)
		# plt.text(1, maxVal-0.3*valRange, "R2: {0:.2f}".format(fomDict['R2'][0]), fontdict=None)
		
		# plt.show()
		
		#plt.text(x, y, s, fontdict=None)
		
		response = input("")
		if response == '+':
			variable += 1
			if variable == targets.shape[2]:
				variable = 0
			skipPlot = True
		elif response == 'x':
			break;
		else:
			skipPlot = False

			
		# if response == '0':
			# variable = 0
		# if response == '1':
			# variable = 1

	return None


# relPointDensity()
#
# This function compter the relative point density for heatmap plots.

def relPointDensity(points, maxDist_rel = 0.1, nonLinMapping = False):

    relPointDensity = np.zeros((points.shape[0], 1))

    # Compute mutual distance matrix for the given point set:
    distMtx = distance_matrix(points, points)
    
    # Set distance threshold relative to the max distance:
    maxDist = maxDist_rel * np.max(distMtx)

    # For each point, count the points closer than the specified 
    # max distance:
    for i in range(points.shape[0]):
        relPointDensity[i] = (distMtx[i,:] < maxDist).sum() - 1
        
    # Scale the relative 'point density' into range [0, 1]:
    scaler = MinMaxScaler()
    scaler.fit(relPointDensity)
    relPointDensity = scaler.transform(relPointDensity)

    # For scaling purposes use optionally non-linear (sqrt) scaling
    # of the densities:
    if nonLinMapping:
        relPointDensity = np.sqrt(relPointDensity)

    return relPointDensity


def scatterPlots(targets, preds, fig = None, ax = None, nrPlotsPerRow = 3, heatScatter = True, plotColor = 'blue', targetVarStr = None, xlabelStr = None, ylabelStr = None, title = None, metadata = None, maxDist_rel = 0.1, nonLinMapping = False, margin_p = [0.1, 0.1], fomBase = [0.05, 0.05], tgtVarBase = [0.3, 0.1], outFile = None, printFoMs = False, fontSizDict = None, axLimits = None):

    #commonCols = ['siteID', 'climID_orig', 'scenario', 'year_start', 'year_end', 'age_pine', 'age_spr', 'age_bl', 'H_pine', 'H_spr', 'H_bl', 'D_pine', 'D_spr', 'D_bl', 'BA_pine', 'BA_spr', 'BA_bl', 'siteType']

    tgtVarUnits = '[$m^3/(ha * y)$]'

    # -----------------------------------------------------

    if fontSizDict is not None:
        fontSiz_xLabel = fontSizDict['fontSiz_xLabel']
        fontSiz_yLabel = fontSizDict['fontSiz_yLabel']
        fontSiz_xTicks = fontSizDict['fontSiz_xTicks']
        fontSiz_yTicks = fontSizDict['fontSiz_yTicks']
        fontSiz_legend = fontSizDict['fontSiz_legend']
    else:
        fontSiz_xLabel = 16
        fontSiz_yLabel = 16
        fontSiz_xTicks = 14
        fontSiz_yTicks = 14
        fontSiz_legend = 14

    if heatScatter:
        points = np.concatenate((targets, preds), axis=1)
        dens = relPointDensity(points, maxDist_rel = maxDist_rel, nonLinMapping = nonLinMapping)
        ax.scatter(targets, preds, s=7, c=dens, vmin=0, vmax=1, cmap = 'turbo')
    else:
        ax.scatter(targets, preds, s=7, c=plotColor)

    # Plot identity line:
    pt = (0, 0)
    ax.axline(pt, slope=1, color='black', linewidth=0.5)
    #plt.plot(targets,targets,'k-') # identity line

    ax.ticklabel_format(useOffset=False)

    N = targets.shape[0]

    
    
    # The next if - else statement should be cleaned!
    if axLimits is None:
        maxValX = max(targets)
        minValX = min(targets)
        valRangeX = maxValX - minValX
        axxMin_x = minValX-margin_p[0]*valRangeX
        axxMax_x = maxValX+margin_p[0]*valRangeX
        
        maxValY = max(preds)
        minValY = min(preds)
        
        valRangeY = maxValY - minValY
        #axxMin_y = minValY-margin_p[1]*valRangeY
        axxMax_y = maxValY+margin_p[1]*valRangeY

        # Force min y-value to zero:
        axxMin_y =0
        #ax.set_ylim(0, axxMax_y) # Plot with margin in y-direction
        #ax.set_xlim(0, axxMax_x) # Plot with margin in x-direction, too
        ax.set_ylim(axxMin_y, axxMax_y) # Plot with margin in y-direction
        ax.set_xlim(axxMin_y, axxMax_x) # Plot with margin in x-direction, too
    else:
        maxValX = axLimits[1]
        minValX = axLimits[0]
        valRangeX = maxValX - minValX
    
        maxValY = axLimits[3]
        minValY = axLimits[2]
        axxMin_y = axLimits[2]
        valRangeY = maxValY - minValY
        #valRangeY = axLimits[3] - axLimits[2]
        
        ax.set_xlim(axLimits[0], axLimits[1]) # Plot with margin in x-direction, too
        ax.set_ylim(axLimits[2], axLimits[3]) # Plot with margin in y-direction

    #ax.legend(fontsize = fontSiz_legend)
    #ax.set_xlabel('Year', fontsize=fontSiz_xLabel)

    #ylabelStr = targetVar.replace('_', ' / ')
    #ylabelStr = ylabelStr.replace('spr', 'spruce')
    #ylabelStr = ylabelStr.replace('bl', 'broadleaved')
    #ylabelStr = ylabelStr + ' ' + tgtVarUnits
    if ylabelStr is not None:
        ax.set_ylabel(ylabelStr, fontsize=fontSiz_yLabel)
    
    # The base (test starting location) in percentages of the whole range:
    base_x = fomBase[0]
    # base_y = from the top of the plot graph:
    base_y = fomBase[1]
    step_y = 0.1

    # -------------------------------------------------------------------------------------
    # NOTE: THE TEXT POSITIONS SHOULD BE RELATIVE TO AXIS MAX (& MIN) VALUES, NOT TO
    # THE DATA MAX & MIN VALUES. PRESENTLY THE CODE IS A MISXTURE OF CONFUSING ASSIGNMENTS!
    # CORRECT!
    # -------------------------------------------------------------------------------------
    if printFoMs:
        fomDict, fomArr = compute_metrics(targets, preds)
        ax.text(base_x*valRangeX, maxValY-base_y*valRangeY, "RMSE%: {0:.1f}".format(fomDict['RMSEp'][0]), fontdict=None)
        ax.text(base_x*valRangeX, maxValY-(base_y+step_y)*valRangeY, "BIAS%: {0:.1f}".format(fomDict['BIASp'][0]), fontdict=None)
        ax.text(base_x*valRangeX, maxValY-(base_y+2*step_y)*valRangeY, "R$^2$: {0:.2f}".format(fomDict['R2'][0]), fontdict=None)
        #ax.text(base_x*valRangeX, maxValY-(base_y+3*step_y)*valRangeY, "mean: {0:.1f}".format(fomDict['y_true_mean'][0]), fontdict=None)

        # Print the target mean value as well: 
        tgtMeanStr = str(round(fomDict['y_true_mean'][0], 1)) 
        ax.text(base_x*valRangeX, maxValY-(base_y+3*step_y)*valRangeY, r"$\bar{x}: $" + tgtMeanStr)

        ax.text(base_x*valRangeX, maxValY-(base_y+4*step_y)*valRangeY, "N: {0:.0f}".format(N), fontdict=None)

        #ax.text(base_x*valRangeX, maxValY-0.05*valRangeY, "RMSE%: {0:.1f}".format(fomDict['RMSEp'][0]), fontdict=None)
        #ax.text(base_x*valRangeX, maxValY-0.15*valRangeY, "BIAS%: {0:.1f}".format(fomDict['BIASp'][0]), fontdict=None)
        #ax.text(base_x*valRangeX, maxValY-0.25*valRangeY, "R2: {0:.2f}".format(fomDict['R2'][0]), fontdict=None)
        #ax.text(base_x*valRangeX, maxValY-0.35*valRangeY, "N: {0:.0f}".format(N), fontdict=None)

    ax.tick_params(axis='x', length=5, direction='in', width=1)
    ax.tick_params(axis='y', length=5, direction='in', width=1)
    #ax.tick_params(axis='x', color='m', length=4, direction='in', width=1,
    #              labelcolor='g', grid_color='b')

    if metadata is not None:
        climId_idx = commonCols.index('climID_orig')
        scenario_idx = commonCols.index('scenario')
        siteType_idx = commonCols.index('siteType')
        age_idx = commonCols.index('age_'+speciesStr)
        H_idx = commonCols.index('H_'+speciesStr)
        D_idx = commonCols.index('D_'+speciesStr)
        BA_idx = commonCols.index('BA_'+speciesStr)
        
        xStartPos = len(tgtTimeSeries)-0.32*len(tgtTimeSeries)
        #ax.text(xStartPos, axxMin_y+0.4*valRangeY, "climID:  {0:.0f}".format(metadata.iloc[ctr,climId_idx]), fontdict=None)
        ax.text(xStartPos, axxMin_y+0.45*valRangeY, "Model: hadgem3_gc31_ll", fontdict=None)
        ax.text(xStartPos, axxMin_y+0.4*valRangeY, "Scenario: {scenario}".format(scenario=metadata.iloc[ctr,scenario_idx]), fontdict=None)
        
        ax.text(xStartPos, axxMin_y+0.3*valRangeY, "Site status @Year: {0:.0f}:".format(axxMin_x-1), fontdict=None)
        ax.text(xStartPos, axxMin_y+0.25*valRangeY, "Age: {0:.0f} $yrs$".format(metadata.iloc[ctr,age_idx]), fontdict=None)
        ax.text(xStartPos, axxMin_y+0.2*valRangeY, "DBH: {0:.0f} $cm$".format(metadata.iloc[ctr,D_idx]), fontdict=None)
        ax.text(xStartPos, axxMin_y+0.15*valRangeY, "Height: {0:.0f} $m$".format(metadata.iloc[ctr,H_idx]), fontdict=None)
        ax.text(xStartPos, axxMin_y+0.1*valRangeY, "Basal area: {0:.0f} $m^2/ha$".format(metadata.iloc[ctr,BA_idx]), fontdict=None)
        ax.text(xStartPos, axxMin_y+0.05*valRangeY, "Fertility class: {0:.0f}".format(metadata.iloc[ctr,siteType_idx]), fontdict=None)
        #if 'H' in targetVar or 'D' in targetVar or 'BA' in targetVar:
        #	targetVarIdx = commonCols.index(targetVar)
        #	ax.text(xStartPos, axxMin_y+0.05*valRangeY, targetVar + ": {0:.0f}".format(metadata.iloc[ctr,targetVarIdx]), fontdict=None)
            
    if xlabelStr is not None:
        ax.set_xlabel(xlabelStr, fontsize=fontSiz_xLabel)

    if targetVarStr is not None:
        tgtVarBase_x = tgtVarBase[0]
        tgtVarBase_y = tgtVarBase[1]
        ax.text(tgtVarBase_x*valRangeX, axxMin_y+tgtVarBase_y*valRangeY, targetVarStr, fontdict=None)
        #ax.text(maxValX-tgtVarBase_x*valRangeX, axxMin_y+tgtVarBase_y*valRangeY, targetVarStr, fontdict=None)

    if title is not None:
        ax.title.set_text(title)
        #fig.suptitle(title, fontsize=14)
        
    if outFile is not None:
        plt.savefig(outFile, dpi=600, format = 'png')
        #plt.savefig(outFile, transparent=True)

    #plt.show()

    return None




# readModelParameters()
#
# This function reads the parameters for model construction, for training, validation
# and test data set generation, and for training the prediction model.
#
# The routine supports two operation modes:
#
# mode = 'CREATE_MODEL'		In this mode the routine reads the parameters for above
#							new model training, and creates an output folder for 
#							saving the results.
#
# mode = 'READ_MODEL'		In this mode the model parameters will be read from an
#							existing model's folder, and the model may be then used
#							for further training, for verification, or for prediction.
#
# Inputs:
#
# paramFilePath				(path) The path (= folder + filename) to the parameter file.
#
# mode						(string) The string defining the operation mode; see mode
#							definitions above. (default = 'CREATE_MODEL')
#
# verbose					(boolean) A flag telling to monitor the read parameters 
#							(default = True).


def readModelParameters(paramFilePath, mode = 'CREATE_MODEL', verbose = True):

	dateTimeNow = dt.now()
	dateStr = dateTimeNow.strftime("%Y%m%d")
	if verbose:
		print("dateStr = ", dateStr)
	
	foo, parameterFileName = os.path.split(paramFilePath)

	paramDict, paramStrDict, pramList = parseModelParams(paramFilePath, verbose = verbose)
	#paramDict, paramStrDict, pramList = parseModelParams(paramFilePath, nullReturn = 'empty', verbose = verbose)

	# Read parameters:
	if mode != 'CREATE_MODEL':
		outPath, fileName = os.path.split(paramFilePath)
		foo, modelIDout = os.path.split(outPath)
		paramDict['saveDataSets'] = False

	# Produce target variable names (add the year string to given variable name(s)):
	targetVarCols = [(i + '_' + str(x+1) + '.0') for i in paramDict['targetVars'] for x in range(paramDict['nYears'])]
	paramDict['targetVarCols'] = targetVarCols
	if verbose:
		print("\ntargetVarCols = \n", targetVarCols)

	# Produce target variable ID string:
	#targetVarStrs = [thisStr.replace('pine', '') for thisStr in paramDict['targetVars']]

	varsList = []
	
	# Leave 'spcStr' out from the 'targetVarIDstr':
	# spcStr = 'sp'

	for ii, thisVarStr in enumerate(paramDict['targetVars']):
		thisVarStr_split = thisVarStr.split('_')
		#print(thisVarStr_split)
		if thisVarStr_split[0] not in varsList:
			varsList.append(thisVarStr_split[0])
			# if (len(thisVarStr_split) > 1) and ii > 0:
				# spcStr += '_'
		# if thisVarStr_split[1] == 'pine':
			# spcStr += '1'
		# if thisVarStr_split[1] == 'spr':
			# spcStr += '2'
		# if thisVarStr_split[1] == 'bl':
			# spcStr += '3'

	#print("varsList = ", varsList)
	# print("spcStr ", spcStr)

	targetVarIDstr=''.join(varsList)
	#targetVarIDstr='_'.join(varsList)
	
	# targetVarIDstr = targetVarIDstr + '_' + spcStr

	#targetVarStrs = [thisStr.replace('_', '') for thisStr in paramDict['targetVars']]
	#targetVarIDstr='_'.join(targetVarStrs)
	if verbose:
		print("\ntargetVarIDstr = ", targetVarIDstr)
	
	# Produce also cascade target variable names (add the year string to given variable name(s)): 
	if paramDict['cascadeTgtVars'] is not None:
		cascadeTgtVars = [(i + '_' + str(x+1) + '.0') for i in paramDict['cascadeTgtVars'] for x in range(paramDict['nYears'])]
		paramDict['cascadeTgtVars'] = cascadeTgtVars
		#print("cascadeTgtVars = ", cascadeTgtVars)

	if paramDict['cascadeInputVars'] is not None:
		cascadeInputVars = [(i + '_' + str(x+1) + '.0') for i in paramDict['cascadeInputVars'] for x in range(paramDict['nYears'])]
		paramDict['cascadeInputVars'] = cascadeInputVars
		#print("cascadeInputVars = ", cascadeInputVars)
		
	if mode == 'CREATE_MODEL':
		# Create model output folder (with optional additional ID string):
		if paramDict['modelType'] == 'XFORMER':
			modelIDout = paramDict['modelType'] + '_' + paramDict['testDefStr'] + '_' + targetVarIDstr + '_' + dateStr
		else:
			modelIDout = paramDict['modelType'] + '_' + paramDict['rnn_type'] + '_' + paramDict['testDefStr'] + '_' + targetVarIDstr + '_' + dateStr
		paramDict['modelIDout'] = modelIDout
		if verbose:
			print("modelID: ", modelIDout)
		
		outPath = os.path.join(paramDict['outPath'], modelIDout)
		paramDict['outPath'] = outPath
		if verbose:
			print("outPath: ", outPath)
		
		if os.path.exists(outPath) == False:
			os.mkdir(outPath)

		# Copy parameter file into output folder:
		parameterFileOut = os.path.join(outPath, parameterFileName)
		copyfile(paramFilePath, parameterFileOut)

		paramDict['saveDataSets'] = True
	else:
		paramDict['saveDataSets'] = False

	# If the data set filters are of type None, then replace them with empty lists:
	paramDict['filters_training'] = [] if paramDict['filters_trainingSet'] == None else paramDict['filters_trainingSet'].split('; ')
	paramDict['filters_valid'] = [] if paramDict['filters_validSet'] == None else paramDict['filters_validSet'].split('; ')
	paramDict['filters_test'] = [] if paramDict['filters_testSet'] == None else paramDict['filters_testSet'].split('; ')

	return paramDict
	

# resultSummary()
#
# This function collects the resuts from different models' fomsPerYear.csv
# files and computes summary statistics. The assumption is, that the target
# variables for each model are equal.
#
# modelListFile		(path = folder + filename) Path to a text file containing the
#					parameter file names (in model folders) of the models to be processed.
#
# fomHdrs			(list of strings) List of the headers of the figures of merit
#					to include in the processing. The default list contains all the
#					headers available in files 'fomsPerYear_*.csv' (* = target variable 
#					string) Default = ['RMSE', 'RMSEp', 'BIAS', 'BIASp', 'R2', 'Ymean']).
#
# summaryHdrs		(list of strings) List of the summary variables to be produced
#					(Default = ['mean', 'std', 'min', 'max']).


def resultSummary(modelListFile, fomHdrs = ['RMSE', 'RMSEp', 'BIAS', 'BIASp', 'R2', 'Ymean'], summaryHdrs = ['mean', 'std', 'min', 'max'], vectorCountFile = None):

	# -------------------------------------------------
	# HARD CODED!
	# Define the species strings:
	speciesStrs = ['pine', 'spr', 'bl']
	# -------------------------------------------------

	outPath, listFileName = os.path.split(modelListFile)
	bar, summaryID = os.path.split(listFileName)
	summaryID = summaryID.split('.')[0]
	#print("summaryID = ", summaryID)
	#print("")

	outputTbl = None

	# Read vector count file, if defined:
	if vectorCountFile is not None:
		vectorCounts = pd.read_csv(vectorCountFile)

	# Define summary dataframe:
	summary_df = pd.DataFrame(index = fomHdrs, columns = summaryHdrs)

	# Read the list of models to process:
	with open(modelListFile) as f:
		modelParamFileList = f.readlines()
		f.close()

	for ii, thisParamFile in enumerate(modelParamFileList):
		# Skip the file if the row commented (starts with '#'):
		if thisParamFile[0] == '#':
			continue

		# Strip possible leading and trailing spaces:
		thisParamFile = thisParamFile.strip()
		
		# foo, fileName = os.path.split(thisParamFile)
		# bar, modelIDout = os.path.split(foo)
		# #print("modelIDout = ", modelIDout)
		# #print("")
		
		# Extract model path:
		modelPath = os.path.dirname(thisParamFile)

		# Read the model parameters:
		paramDict = readModelParameters(thisParamFile, mode = 'READ_MODEL', verbose = False)

		# Get model target variables:
		targetVars = paramDict['targetVars']

		# Define summary dataframe:
		outputTblCols = [(x + '_' + i) for i in speciesStrs for x in fomHdrs]
		outputTblCols = ['Variable'] + ['modelType'] + ['rnnType'] + ['dataPerc'] + ['climDataPeriod'] + ['ModelPath'] + outputTblCols
		
		#outputTblCols = [(x + '_' + i) for i in targetVars for x in fomHdrs]
		#outputTblCols = outputTblCols + ['modelType'] + ['rnnType'] + ['dataPerc'] + ['climDataPeriod'] + ['ModelPath']
		#print("outputTblCols = ", outputTblCols)
		
		#summary_H_df = pd.DataFrame(columns = outputTblCols, index = summaryHdrs)
		
		#print("summary_H_df.shape = ", summary_H_df.shape)
		#summary_df = pd.DataFrame(index = fomHdrs, columns = summaryHdrs)

		if paramDict['modelType'] == 'XFORMER':
			rnnType = 'NA'
		else:
			rnnType = paramDict['rnn_type']
			
		if paramDict['input_dim_enc'] == 12:
			climDataPeriod = 'Yearly'
		elif paramDict['input_dim_enc'] == 144:
			climDataPeriod = 'Monthly'
		else:
			climDataPeriod = 'BI-Monthly'
			
		if vectorCountFile is not None:
			if paramDict['filters_trainingSet'].split(';')[0].split(' ')[0] == 'runID':
				# Search the amount of training data used (%) from table!
				compOperator = paramDict['filters_trainingSet'].split(';')[0].split(' ')[1]
				runID_limit = int(paramDict['filters_trainingSet'].split(';')[0].split(' ')[2])
				
				# NOTE: Presently the extraction of the input data relative size (percentage)
				# wrt. to the total input data size is restricted to the cases where the 
				# size is limited by the 'runID' column of the input data table, and with
				# the filter operators '<' and '<=' only!
				if compOperator == '<=':
					vectorCounts_lim = vectorCounts.loc[vectorCounts['runID'] <= runID_limit]
					dataPerc = vectorCounts_lim['CumPerc'].values[-1]
				else:
					vectorCounts_lim = vectorCounts.loc[vectorCounts['runID'] < runID_limit]
					dataPerc = vectorCounts_lim['CumPerc'].values[-1]
		else:
			dataPerc = np.nan

		# Count for three species per variable:
		speciesCount = 0

		# Read the fomsPerYear_*.csv tables:
		for thisTgtVar in targetVars:
			summary_df = pd.DataFrame(index = fomHdrs, columns = summaryHdrs)
			if speciesCount == 0:
				summary_H_df = pd.DataFrame(columns = outputTblCols, index = summaryHdrs)
		
			# Strip the species string from the variable name:
			thisVariable = thisTgtVar[0:thisTgtVar.find('_')]
		
			thisVarFomsPerYearFile = os.path.join(modelPath, 'fomsPerYear_' + thisTgtVar + '.csv')
			thisVarFomsPerYear = pd.read_csv(thisVarFomsPerYearFile)

			#print("thisTgtVar = ", thisTgtVar)
			#print("thisVariable = ", thisVariable)
			summary_df['std'] = thisVarFomsPerYear[fomHdrs].std()
			summary_df['min'] = thisVarFomsPerYear[fomHdrs].min()
			summary_df['max'] = thisVarFomsPerYear[fomHdrs].max()

			# Compute mean values (mean absolute value for bias & bias%.
			# Before taking the mean, convert to absolute values. Only bias
			# contains also negative values, so there is no change to other
			# figures of merit.
			thisVarFomsPerYear[fomHdrs] = thisVarFomsPerYear[fomHdrs].abs()
			summary_df['mean'] = thisVarFomsPerYear[fomHdrs].mean()
			
			if 'pine' in thisTgtVar:
				thisSpecies = 'pine'
				speciesCount += 1
			if 'spr' in thisTgtVar:
				thisSpecies = 'spr'
				speciesCount += 1
			if 'bl' in thisTgtVar:
				thisSpecies = 'bl'
				speciesCount += 1

			thisVarHdrs = [(y + '_' + thisSpecies) for y in fomHdrs]
			#thisVarHdrs = [(y + '_' + thisTgtVar) for y in fomHdrs]
			#print("thisVarHdrs = ", thisVarHdrs)

			#print("")
			#print(summary_df['mean'].values.flatten(order='F'))
			
			summary_H_df.loc['mean', thisVarHdrs] = summary_df['mean'].values.flatten(order='F')
			summary_H_df.loc['std', thisVarHdrs] = summary_df['std'].values.flatten(order='F')
			summary_H_df.loc['min', thisVarHdrs] = summary_df['min'].values.flatten(order='F')
			summary_H_df.loc['max', thisVarHdrs] = summary_df['max'].values.flatten(order='F')

			for thisSHdr in summaryHdrs:
				summary_H_df.loc[thisSHdr, 'Variable'] = thisVariable
				
			for thisSHdr in summaryHdrs:
				summary_H_df.loc[thisSHdr, 'modelType'] = paramDict['modelType']
				
			for thisSHdr in summaryHdrs:
				summary_H_df.loc[thisSHdr, 'rnnType'] = rnnType
				
			for thisSHdr in summaryHdrs:
				summary_H_df.loc[thisSHdr, 'dataPerc'] = dataPerc
				
			for thisSHdr in summaryHdrs:
				summary_H_df.loc[thisSHdr, 'climDataPeriod'] = climDataPeriod
				
			for thisSHdr in summaryHdrs:
				summary_H_df.loc[thisSHdr, 'ModelPath'] = modelPath
			
				
			#print(summary_df) 
			
			#print(thisVarFomsPerYear[fomHdrs].mean())
			#print(thisVarFomsPerYear[fomHdrs].std())
			#print(thisVarFomsPerYear[fomHdrs].min())
			#print(thisVarFomsPerYear[fomHdrs].max())
			#print("")
			
			# When all species (total = 3) of this variable has peen added to
			# summary_H_df, then concatenate the results to output table:			
			if speciesCount == 3:
				outputTbl = pd.concat([outputTbl, summary_H_df], axis=0)
				speciesCount = 0
				
				#print("thisVariable = ", thisVariable)
				#print("")
				#print(outputTbl) 
				#print("")
				#print("")
				
		#outputTbl = pd.concat([outputTbl, summary_H_df], axis=0)

	outFile = os.path.join(outPath, summaryID + '_summary.csv')
	outputTbl.to_csv(outFile, sep = ',', index = True)

	print("summary table saved into: ", outFile)
		
	return outputTbl


def resultSummary_multiVar(modelListFile, fomHdrs = ['RMSE', 'RMSEp', 'BIAS', 'BIASp', 'R2', 'Ymean'], summaryHdrs = ['mean', 'meanabs', 'std', 'min', 'max'], fomType = 'fomsPerYear', vectorCountFile = None):

    outPath, listFileName = os.path.split(modelListFile)
    bar, summaryID = os.path.split(listFileName)
    summaryID = summaryID.split('.')[0]
    #print("summaryID = ", summaryID)
    #print("")

    outputTbl = None

    # Read vector count file, if defined:
    if vectorCountFile is not None:
        vectorCounts = pd.read_csv(vectorCountFile)

    # Read the list of models to process:
    with open(modelListFile) as f:
        modelParamFileList = f.readlines()
        f.close()

    for ii, thisParamFile in enumerate(modelParamFileList):
        # Skip the file if the row commented (starts with '#'):
        if thisParamFile[0] == '#':
            continue

        # Strip possible leading and trailing spaces:
        thisParamFile = thisParamFile.strip()
        
        # Extract model path:
        modelPath = os.path.dirname(thisParamFile)

        # Read the model parameters:
        paramDict = readModelParameters(thisParamFile, mode = 'READ_MODEL', verbose = False)

        # Get model target variables:
        targetVars = paramDict['targetVars']

        # The summary headers for the output dataframe:
        fomSumHdrs = [(y + '_' + i) for i in summaryHdrs for y in fomHdrs]
        #print("fomSumHdrs = ", fomSumHdrs)

        outputTblCols = ['modelType'] + ['rnnType'] + ['dataPerc'] + ['tgtVar'] + ['climDataPeriod'] + fomSumHdrs + ['ModelPath']
        #print("outputTblCols = ", outputTblCols)

        summary_H_df = pd.DataFrame(columns = outputTblCols, index = targetVars)
        
        if paramDict['modelType'] == 'XFORMER':
            rnnType = 'NA'
        else:
            rnnType = paramDict['rnn_type']
            
        if paramDict['input_dim_enc'] == 12:
            climDataPeriod = 'Yearly'
        elif paramDict['input_dim_enc'] == 144:
            climDataPeriod = 'Monthly'
        else:
            climDataPeriod = 'BI-Monthly'
            
        if vectorCountFile is not None:
            if paramDict['filters_trainingSet'].split(';')[0].split(' ')[0] == 'runID':
                # Search the amount of training data used (%) from table!
                compOperator = paramDict['filters_trainingSet'].split(';')[0].split(' ')[1]
                runID_limit = int(paramDict['filters_trainingSet'].split(';')[0].split(' ')[2])
                
                # NOTE: Presently the extraction of the input data relative size (percentage)
                # wrt. to the total input data size is restricted to the cases where the 
                # size is limited by the 'runID' column of the input data table, and with
                # the filter operators '<' and '<=' only!
                if compOperator == '<=':
                    vectorCounts_lim = vectorCounts.loc[vectorCounts['runID'] <= runID_limit]
                    dataPerc = vectorCounts_lim['CumPerc'].values[-1]
                else:
                    vectorCounts_lim = vectorCounts.loc[vectorCounts['runID'] < runID_limit]
                    dataPerc = vectorCounts_lim['CumPerc'].values[-1]
        else:
            dataPerc = np.nan

        # Read the input tables (fomsPerYear_*.csv or fomsPerCase_*.csv)
        # and compute the summary statistics for each target variable:
        for thisTgtVar in targetVars:
            #print("thisTgtVar = ", thisTgtVar)
            
            thisVarFomsFile = os.path.join(modelPath, fomType + '_' + thisTgtVar + '.csv')
            thisVarFoms = pd.read_csv(thisVarFomsFile)
            
            # If the option fomType = 'fomsPerCase' was selected, then remove
            # rows that contain NaN's:
            if fomType == 'fomsPerCase':
                thisVarFoms = thisVarFoms.dropna()

            # Define summary dataframe (initialize to empty dataframe):
            summary_df = pd.DataFrame()

            # Compute the desired summary statistics from the original
            # set of figures of merit (read from the input file):
            for thisSummaryItem in summaryHdrs:
                if thisSummaryItem == 'mean':
                    summary_df = pd.concat([summary_df, thisVarFoms[fomHdrs].mean()], axis=1)
                if thisSummaryItem == 'std':
                    summary_df = pd.concat([summary_df, thisVarFoms[fomHdrs].std()], axis=1)
                if thisSummaryItem == 'min':
                    summary_df = pd.concat([summary_df, thisVarFoms[fomHdrs].min()], axis=1)
                if thisSummaryItem == 'max':
                    summary_df = pd.concat([summary_df, thisVarFoms[fomHdrs].max()], axis=1)
                if thisSummaryItem == 'meanabs':
                    # Compute the mean of absolute values of bias & bias%.
                    # Only bias & bias% may contain negative values, so there 
                    # is no effect to other figures of merit, so the *_meanabs
                    # results for other figures of merit are redundant to *_mean (lazy coding).
                    thisVarFoms_abs = thisVarFoms[fomHdrs].abs()
                    summary_df = pd.concat([summary_df, thisVarFoms_abs.mean()], axis=1)

            #summary_df = summary_df.T
            summary_df = summary_df.unstack()

            #print("")
            #print("summary_df = ")
            #print(summary_df)

            summary_H_df.loc[thisTgtVar, fomSumHdrs] = summary_df.values

            for thisSHdr in summaryHdrs:
                summary_H_df.loc[thisTgtVar, 'modelType'] = paramDict['modelType']
                
            for thisSHdr in summaryHdrs:
                summary_H_df.loc[thisTgtVar, 'rnnType'] = rnnType
                
            for thisSHdr in summaryHdrs:
                summary_H_df.loc[thisTgtVar, 'dataPerc'] = dataPerc
            
            for thisSHdr in summaryHdrs:
                summary_H_df.loc[thisTgtVar, 'tgtVar'] = thisTgtVar

            for thisSHdr in summaryHdrs:
                summary_H_df.loc[thisTgtVar, 'climDataPeriod'] = climDataPeriod
                
            for thisSHdr in summaryHdrs:
                summary_H_df.loc[thisTgtVar, 'ModelPath'] = modelPath
        
        # Concatenate the summary FoM's of this target variable to the
        # output dataframe.
        outputTbl = pd.concat([outputTbl, summary_H_df], axis=0)

    outFile = os.path.join(outPath, summaryID + '_' + fomType + '_summary.csv')
    outputTbl.to_csv(outFile, sep = ',', index = True)

    print("summary table saved into: ", outFile)
        
    return outputTbl


def resultSummary_multiVar_ori(modelListFile, fomHdrs = ['RMSE', 'RMSEp', 'BIAS', 'BIASp', 'R2', 'Ymean'], summaryHdrs = ['mean', 'meanabs', 'std', 'min', 'max'], vectorCountFile = None):


	outPath, listFileName = os.path.split(modelListFile)
	bar, summaryID = os.path.split(listFileName)
	summaryID = summaryID.split('.')[0]
	#print("summaryID = ", summaryID)
	#print("")

	outputTbl = None

	# Read vector count file, if defined:
	if vectorCountFile is not None:
		vectorCounts = pd.read_csv(vectorCountFile)

	# Define summary dataframe:
	summary_df = pd.DataFrame(index = fomHdrs, columns = summaryHdrs)

	# Read the list of models to process:
	with open(modelListFile) as f:
		modelParamFileList = f.readlines()
		f.close()

	for ii, thisParamFile in enumerate(modelParamFileList):
		# Skip the file if the row commented (starts with '#'):
		if thisParamFile[0] == '#':
			continue

		# Strip possible leading and trailing spaces:
		thisParamFile = thisParamFile.strip()
		
		# foo, fileName = os.path.split(thisParamFile)
		# bar, modelIDout = os.path.split(foo)
		# #print("modelIDout = ", modelIDout)
		# #print("")
		
		# Extract model path:
		modelPath = os.path.dirname(thisParamFile)

		# Read the model parameters:
		paramDict = readModelParameters(thisParamFile, mode = 'READ_MODEL', verbose = False)

		# Get model target variables:
		targetVars = paramDict['targetVars']

		# Define summary dataframe:
		outputTblCols = [(x + '_' + i) for i in targetVars for x in fomHdrs]
		outputTblCols = outputTblCols + ['modelType'] + ['rnnType'] + ['dataPerc'] + ['climDataPeriod'] + ['ModelPath']
		#print("outputTblCols = ", outputTblCols)
		summary_H_df = pd.DataFrame(columns = outputTblCols, index = summaryHdrs)
		
		#print("summary_H_df.shape = ", summary_H_df.shape)
		#summary_df = pd.DataFrame(index = fomHdrs, columns = summaryHdrs)

		if paramDict['modelType'] == 'XFORMER':
			rnnType = 'NA'
		else:
			rnnType = paramDict['rnn_type']
			
		if paramDict['input_dim_enc'] == 12:
			climDataPeriod = 'Yearly'
		elif paramDict['input_dim_enc'] == 144:
			climDataPeriod = 'Monthly'
		else:
			climDataPeriod = 'BI-Monthly'
			
		if vectorCountFile is not None:
			if paramDict['filters_trainingSet'].split(';')[0].split(' ')[0] == 'runID':
				# Search the amount of training data used (%) from table!
				compOperator = paramDict['filters_trainingSet'].split(';')[0].split(' ')[1]
				runID_limit = int(paramDict['filters_trainingSet'].split(';')[0].split(' ')[2])
				
				# NOTE: Presently the extraction of the input data relative size (percentage)
				# wrt. to the total input data size is restricted to the cases where the 
				# size is limited by the 'runID' column of the input data table, and with
				# the filter operators '<' and '<=' only!
				if compOperator == '<=':
					vectorCounts_lim = vectorCounts.loc[vectorCounts['runID'] <= runID_limit]
					dataPerc = vectorCounts_lim['CumPerc'].values[-1]
				else:
					vectorCounts_lim = vectorCounts.loc[vectorCounts['runID'] < runID_limit]
					dataPerc = vectorCounts_lim['CumPerc'].values[-1]
		else:
			dataPerc = np.nan

		# Read the fomsPerYear_*.csv tables:
		for thisTgtVar in targetVars:
			thisVarFomsPerYearFile = os.path.join(modelPath, 'fomsPerYear_' + thisTgtVar + '.csv')
			thisVarFomsPerYear = pd.read_csv(thisVarFomsPerYearFile)

			#print("thisTgtVar = ", thisTgtVar)
			summary_df['std'] = thisVarFomsPerYear[fomHdrs].std()
			summary_df['min'] = thisVarFomsPerYear[fomHdrs].min()
			summary_df['max'] = thisVarFomsPerYear[fomHdrs].max()

			summary_df['mean'] = thisVarFomsPerYear[fomHdrs].mean()

			# Compute mean values (mean absolute value for bias & bias%.
			# Before taking the mean, convert to absolute values. Only bias
			# contains also negative values, so there is no change to other
			# figures of merit.
			thisVarFomsPerYear[fomHdrs] = thisVarFomsPerYear[fomHdrs].abs()
			summary_df['meanabs'] = thisVarFomsPerYear[fomHdrs].mean()

			thisVarHdrs = [(y + '_' + thisTgtVar) for y in fomHdrs]
			#print("thisVarHdrs = ", thisVarHdrs)

			#print("")
			#print("summary_df['mean'].values.flatten(order='F') = ")
			#print(summary_df['mean'].values.flatten(order='F'))
			
			summary_H_df.loc['mean', thisVarHdrs] = summary_df['mean'].values.flatten(order='F')
			summary_H_df.loc['meanabs', thisVarHdrs] = summary_df['meanabs'].values.flatten(order='F')
			summary_H_df.loc['std', thisVarHdrs] = summary_df['std'].values.flatten(order='F')
			summary_H_df.loc['min', thisVarHdrs] = summary_df['min'].values.flatten(order='F')
			summary_H_df.loc['max', thisVarHdrs] = summary_df['max'].values.flatten(order='F')

			for thisSHdr in summaryHdrs:
				summary_H_df.loc[thisSHdr, 'modelType'] = paramDict['modelType']
				
			for thisSHdr in summaryHdrs:
				summary_H_df.loc[thisSHdr, 'rnnType'] = rnnType
				
			for thisSHdr in summaryHdrs:
				summary_H_df.loc[thisSHdr, 'dataPerc'] = dataPerc
				
			for thisSHdr in summaryHdrs:
				summary_H_df.loc[thisSHdr, 'climDataPeriod'] = climDataPeriod
				
			for thisSHdr in summaryHdrs:
				summary_H_df.loc[thisSHdr, 'ModelPath'] = modelPath
			
				
			print("")
			print("summary_df = ") 
			print(summary_df)
            
            # 28.8.2024:
            # Transpose the table 'summary_df' and add 
			
			#print("")
			#print(thisVarFomsPerYear[fomHdrs].mean())
			#print(thisVarFomsPerYear[fomHdrs].std())
			#print(thisVarFomsPerYear[fomHdrs].min())
			#print(thisVarFomsPerYear[fomHdrs].max())
			#print("")

		outputTbl = pd.concat([outputTbl, summary_H_df], axis=0)

	outFile = os.path.join(outPath, summaryID + '_summary.csv')
	outputTbl.to_csv(outFile, sep = ',', index = True)

	print("summary table saved into: ", outFile)
		
	return outputTbl
	
	
# ==========================================================================
# epoch_time() = a function to show how long an epoch takes.
# --------------------------------------------------------------------------
def elapsedTime(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
