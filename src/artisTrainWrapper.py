# artisTrainWrapper.py
#
# This is a stand-alone version of the code originally written in Seq2seq_model.ipynb
#
# The purpose is to train + verify a PyTorch model from command line
# interface and in Linux operation system to performa a grid search
# for model hyperparameters.

import os
import datetime
import sys
import math
import random
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from platform import python_version

from gen_utils_art import readModelParameters
from gen_utils_art import predsdeNorm, computeResults, joinTestDataWithResults
from seq2seq_model import initModel

from pytorchtools import constructDataSets
from pytorchtools import evaluate, trainValidPredictions
from pytorchtools import trainingLoop

from gen_utils_art import elapsedTime

print("")
print("Current Python Version -", python_version())
print("Pandas version -", pd.__version__)


# ==========================================================================
# searchWrapper_FC_RNN()
#
# This function runs the model training with a set of input parameter 
# combinations that have been defined in paramDict. The output for ranking
# the different combination is the best validation loss (bestValidLoss)
# recorded during the model training. The parameter combibations and the
# resulting validation loss will be returned in Pandas DataFrame 'wrapperTbl'
# which will alos be written into a *.csv file, if output file name will be
# provided.
# --------------------------------------------------------------------------

def searchWrapper_FC_RNN(paramDict, training_loader, validation_loader, outFile = None, verbose = False):

	rnnTypes = paramDict['rnnTypes']
	nrEncLayerss = paramDict['nrEncLayerss']
	encHidDimss = paramDict['encHidDimss']
	nrFc2h0Layerss = paramDict['nrFc2h0Layerss']
	nrFcHidLrs = paramDict['nrFcHidLrs']
	
	learningRates = paramDict['learningRates']
	batchSizes = paramDict['batchSizes']
	encDropouts = paramDict['encDropouts']
	fcDropouts = paramDict['fcDropouts']

	# Initialize output DataFrame:
	nrTbRows = len(rnnTypes)*len(nrEncLayerss)*len(encHidDimss)*len(nrFc2h0Layerss)*len(nrFcHidLrs)*len(learningRates)*len(batchSizes)*len(encDropouts)*len(fcDropouts)
	colHdrs = ['modelType','rnn_type','n_layers_enc','hid_dim_enc','n_layers_fc2h0','nr_hid_fc','learning_rate','batchSize','dropout_enc','dropout_fc', 'trainLoss', 'validLoss']
	wrapperTbl = pd.DataFrame(index = range(nrTbRows), columns = colHdrs)

	if verbose:
		print("modelType = ", paramDict['modelType'])
		print("rnnTypes = ", rnnTypes)
		print("nrEncLayerss = ", nrEncLayerss)
		print("encHidDimss = ", encHidDimss)
		print("nrFc2h0Layerss = ", nrFc2h0Layerss)
		print("nrFcHidLrs = ", nrFcHidLrs)
		
		print("learningRates = ", learningRates)
		print("batchSizes = ", batchSizes)
		print("encDropouts = ", encDropouts)
		print("fcDropouts = ", fcDropouts)
		print("wrapperTbl.shape = ", wrapperTbl.shape)

	tblCtr = 0

	for thisRnnType in rnnTypes:
		paramDict['rnn_type'] = thisRnnType

		for thisNrEcLayers in nrEncLayerss:
			paramDict['n_layers_enc'] = thisNrEcLayers
			
			for thisEncHidDim in encHidDimss:
				paramDict['hid_dim_enc'] = thisEncHidDim

				for thisNrFc2h0Layers in nrFc2h0Layerss:
					paramDict['n_layers_fc2h0'] = thisNrFc2h0Layers
					
					for thisNrFcHidLrs in nrFcHidLrs:
						paramDict['nr_hid_fc'] = thisNrFcHidLrs
						paramDict['fc_in_sizes'] = paramDict['fc_in_sizes'][0:thisNrFcHidLrs]
						
						for thisLearningRate in learningRates:
							paramDict['learning_rate'] = thisLearningRate

							for thisBatchSize in batchSizes:
								paramDict['batchSize'] = thisBatchSize
								
								for thisEncDropout in encDropouts:
									paramDict['dropout_enc'] = thisEncDropout
			
									for thisFcDropout in fcDropouts:
										paramDict['dropout_fc'] = thisFcDropout
										
										print("rnn_type = ", thisRnnType)
										print("n_layers_enc = ", thisNrEcLayers)
										print("hid_dim_enc = ", thisEncHidDim)
										print("n_layers_fc2h0 = ", thisNrFc2h0Layers)
										print("nr_hid_fc = ", thisNrFcHidLrs)
										print("learning_rate = ", thisLearningRate)
										print("batchSize = ", thisBatchSize)
										print("dropout_enc = ", thisEncDropout)
										print("dropout_fc = ", thisFcDropout)
			
										wrapperTbl['modelType'][tblCtr] = paramDict['modelType']
										wrapperTbl['rnn_type'][tblCtr] = thisRnnType
										wrapperTbl['n_layers_enc'][tblCtr] = thisNrEcLayers
										wrapperTbl['hid_dim_enc'][tblCtr] = thisEncHidDim
										wrapperTbl['n_layers_fc2h0'][tblCtr] = thisNrFc2h0Layers
										wrapperTbl['nr_hid_fc'][tblCtr] = thisNrFcHidLrs
										wrapperTbl['learning_rate'][tblCtr] = thisLearningRate
										wrapperTbl['batchSize'][tblCtr] = thisBatchSize
										wrapperTbl['dropout_enc'][tblCtr] = thisEncDropout
										wrapperTbl['dropout_fc'][tblCtr] = thisFcDropout
			
										# Model definition
										model, criterion, optimizer, scheduler = initModel(paramDict)
										
										# Call trainingLoop() with the defined model and the given parameters:
										model, trainvalidLosses, bestValidLoss, monTrainLoss = trainingLoop(model, paramDict, training_loader, validation_loader, optimizer, criterion, verbose = False, monAllEpochs = True)
			
										wrapperTbl['trainLoss'][tblCtr] = monTrainLoss
										wrapperTbl['validLoss'][tblCtr] = bestValidLoss
										tblCtr += 1

										if outFile is not None:
											wrapperTbl.to_csv(outFile, sep = ',', index = False)

	return wrapperTbl


# ==========================================================================
# searchWrapperS2S()
#
# Parameter search wrapper for seq2seq architecture.
# --------------------------------------------------------------------------
def searchWrapperS2S(paramDict, training_loader, validation_loader, outFile = None, verbose = False):

	rnnTypes = paramDict['rnnTypes']
	nrEncLayerss = paramDict['nrEncLayerss']
	encHidDimss = paramDict['encHidDimss']
	#nrFc2h0Layerss = paramDict['nrFc2h0Layerss']
	nrFcHidLrs = paramDict['nrFcHidLrs']
	
	learningRates = paramDict['learningRates']
	batchSizes = paramDict['batchSizes']
	encDropouts = paramDict['encDropouts']
	fcDropouts = paramDict['fcDropouts']
	dropoutDecs = paramDict['dropoutDecs']
	teacherForcingRatios = paramDict['teacherForcingRatios']

	# Initialize output DataFrame:
	nrTbRows = len(rnnTypes)*len(nrEncLayerss)*len(encHidDimss)*len(nrFcHidLrs)*len(learningRates)*len(batchSizes)*len(encDropouts)*len(fcDropouts)*len(dropoutDecs)*len(teacherForcingRatios)
	colHdrs = ['modelType','rnn_type','n_layers_enc','hid_dim_enc','nr_hid_fc','learning_rate','batchSize','dropout_enc','dropout_fc', 'dropout_dec', 'teacher_forcing_ratio', 'trainLoss', 'validLoss']
	wrapperTbl = pd.DataFrame(index = range(nrTbRows), columns = colHdrs)

	if verbose:
		print("modelType = ", paramDict['modelType'])
		print("rnnTypes = ", rnnTypes)
		print("nrEncLayerss = ", nrEncLayerss)
		print("encHidDimss = ", encHidDimss)
		print("nrFcHidLrs = ", nrFcHidLrs)
		
		print("learningRates = ", learningRates)
		print("batchSizes = ", batchSizes)
		print("encDropouts = ", encDropouts)
		print("fcDropouts = ", fcDropouts)
		print("dropoutDecs = ", dropoutDecs)
		print("teacherForcingRatios = ", teacherForcingRatios)
		print("wrapperTbl.shape = ", wrapperTbl.shape)

	tblCtr = 0

	for thisRnnType in rnnTypes:
		paramDict['rnn_type'] = thisRnnType

		for thisNrEcLayers in nrEncLayerss:
			paramDict['n_layers_enc'] = thisNrEcLayers
			
			for thisEncHidDim in encHidDimss:
				paramDict['hid_dim_enc'] = thisEncHidDim

				for thisNrFcHidLrs in nrFcHidLrs:
					paramDict['nr_hid_fc'] = thisNrFcHidLrs
					paramDict['fc_in_sizes'] = paramDict['fc_in_sizes'][0:thisNrFcHidLrs]
					
					for thisLearningRate in learningRates:
						paramDict['learning_rate'] = thisLearningRate

						for thisBatchSize in batchSizes:
							paramDict['batchSize'] = thisBatchSize
							
							for thisEncDropout in encDropouts:
								paramDict['dropout_enc'] = thisEncDropout
		
								for thisFcDropout in fcDropouts:
									paramDict['dropout_fc'] = thisFcDropout
		
									for thisDecDropout in dropoutDecs:
										paramDict['dropout_dec'] = thisDecDropout
										
										for thisTeacherForcingRatio in teacherForcingRatios:
											paramDict['teacher_forcing_ratio'] = thisTeacherForcingRatio
											
											print("rnn_type = ", thisRnnType)
											print("n_layers_enc = ", thisNrEcLayers)
											print("hid_dim_enc = ", thisEncHidDim)
											print("nr_hid_fc = ", thisNrFcHidLrs)
											print("learning_rate = ", thisLearningRate)
											print("batchSize = ", thisBatchSize)
											print("dropout_enc = ", thisEncDropout)
											print("dropout_fc = ", thisFcDropout)
											print("dropout_dec = ", thisDecDropout)
											print("teacher_forcing_ratio = ", thisTeacherForcingRatio)
										
											wrapperTbl['modelType'][tblCtr] = paramDict['modelType']
											wrapperTbl['rnn_type'][tblCtr] = thisRnnType
											wrapperTbl['n_layers_enc'][tblCtr] = thisNrEcLayers
											wrapperTbl['hid_dim_enc'][tblCtr] = thisEncHidDim
											wrapperTbl['nr_hid_fc'][tblCtr] = thisNrFcHidLrs
											wrapperTbl['learning_rate'][tblCtr] = thisLearningRate
											wrapperTbl['batchSize'][tblCtr] = thisBatchSize
											wrapperTbl['dropout_enc'][tblCtr] = thisEncDropout
											wrapperTbl['dropout_fc'][tblCtr] = thisFcDropout
											wrapperTbl['dropout_dec'][tblCtr] = thisDecDropout
											wrapperTbl['teacher_forcing_ratio'][tblCtr] = thisTeacherForcingRatio
				
											# Model definition
											model, criterion, optimizer, scheduler = initModel(paramDict)
											
											# Call trainingLoop() with the defined model and the given parameters:
											model, trainvalidLosses, bestValidLoss, monTrainLoss = trainingLoop(model, paramDict, training_loader, validation_loader, optimizer, criterion, verbose = False, monAllEpochs = True)
				
											wrapperTbl['trainLoss'][tblCtr] = monTrainLoss
											wrapperTbl['validLoss'][tblCtr] = bestValidLoss
											tblCtr += 1

											if outFile is not None:
												wrapperTbl.to_csv(outFile, sep = ',', index = False)

	return wrapperTbl



# ==========================================================================
# searchWrapperTF()
#
# Parameter search wrapper for transformer architecture.
# --------------------------------------------------------------------------
def searchWrapperTF(paramDict, training_loader, validation_loader, outFile = None, verbose = False):

	tfNheads = paramDict['tfNheads']
	tfHidDims = paramDict['tfHidDims']
	tfNlayerss = paramDict['tfNlayerss']
	
	learningRates = paramDict['learningRates']
	batchSizes = paramDict['batchSizes']
	tfDropouts = paramDict['tfDropouts']
	
	# Initialize output DataFrame:
	nrTbRows = len(tfNheads)*len(tfHidDims)*len(tfNlayerss)*len(learningRates)*len(batchSizes)*len(tfDropouts)
	colHdrs = ['modelType','nhead_tf','hid_dim_tf','nlayers_tf','learning_rate','batchSize','dropout_tf','trainLoss', 'validLoss']
	wrapperTbl = pd.DataFrame(index = range(nrTbRows), columns = colHdrs)

	if verbose:
		print("modelType = ", paramDict['modelType'])
		print("tfNheads = ", tfNheads)
		print("tfHidDims = ", tfHidDims)
		print("tfNlayerss = ", tfNlayerss)
		
		print("learningRates = ", learningRates)
		print("batchSizes = ", batchSizes)
		print("tfDropouts = ", tfDropouts)

	tblCtr = 0

	for thisTfNheads in tfNheads:
		paramDict['nhead_tf'] = thisTfNheads

		for thisTfHidDim in tfHidDims:
			paramDict['hid_dim_tf'] = thisTfHidDim

			for thisTfNlayers in tfNlayerss:
				paramDict['nlayers_tf'] = thisTfNlayers
				
				for thisLearningRate in learningRates:
					paramDict['learning_rate'] = thisLearningRate

					for thisBatchSize in batchSizes:
						paramDict['batchSize'] = thisBatchSize
						
						for thisTfDropout in tfDropouts:
							paramDict['dropout_tf'] = thisTfDropout

							print("nhead_tf = ", thisTfNheads)
							print("hid_dim_tf = ", thisTfHidDim)
							print("nlayers_tf = ", thisTfNlayers)
							
							print("learning_rate = ", thisLearningRate)
							print("batchSize = ", thisBatchSize)
							print("dropout_tf = ", thisTfDropout)
	
							wrapperTbl['modelType'][tblCtr] = paramDict['modelType']
							wrapperTbl['nhead_tf'][tblCtr] = thisTfNheads
							wrapperTbl['hid_dim_tf'][tblCtr] = thisTfHidDim
							wrapperTbl['nlayers_tf'][tblCtr] = thisTfNlayers
							
							wrapperTbl['learning_rate'][tblCtr] = thisLearningRate
							wrapperTbl['batchSize'][tblCtr] = thisBatchSize
							wrapperTbl['dropout_tf'][tblCtr] = thisTfDropout

							# Model definition
							model, criterion, optimizer, scheduler = initModel(paramDict)
							
							# Call trainingLoop() with the defined model and the given parameters:
							model, trainvalidLosses, bestValidLoss, monTrainLoss = trainingLoop(model, paramDict, training_loader, validation_loader, optimizer, criterion, verbose = False, monAllEpochs = True)

							wrapperTbl['trainLoss'][tblCtr] = monTrainLoss
							wrapperTbl['validLoss'][tblCtr] = bestValidLoss
							tblCtr += 1

							if outFile is not None:
								wrapperTbl.to_csv(outFile, sep = ',', index = False)

	return wrapperTbl




if __name__ == '__main__':

	startTime = time.time()

	# ------------------------------------------------------------
	# Argument handling
	# ------------------------------------------------------------
	parser = argparse.ArgumentParser()
	parser.add_argument('--param_file', '-p', required=True, help='Parameter file')

	args = parser.parse_args()

	# Print the arguments
	for key in vars(args):
		print('\t' + key + ' = ' + str(getattr(args, key)))

	# ------------------------------------------------------------
	# Read parameters from text file:
	# ------------------------------------------------------------
	print("1. Read parameters & init the model...")
	paramDict = readModelParameters(args.param_file, mode = 'CREATE_MODEL', verbose = True)

	# ------------------------------------------------------------
	# Construct Datasets
	# ------------------------------------------------------------
	print("2. Construct data sets ...")
	start_time = time.time()
	training_loader, validation_loader, test_loader, dataset_train, dataset_valid, dataset_test = constructDataSets(paramDict)
	end_time = time.time()
	t_mins, t_secs = elapsedTime(start_time, end_time)
	print("Building data sets took: {:02} min {:02} sec \r".format(t_mins, t_secs), end="")

	# ------------------------------------------------------------
	# Model training wrapper
	# ------------------------------------------------------------
	print("3. Start the model wrappper ...")
	
	foo, filename = os.path.split(args.param_file)
	ff, ext = os.path.splitext(filename)
	outFile = os.path.join(paramDict['outPath'], ff + '_wrapperOut.csv')

	if paramDict['modelType'] == 'FC_RNN':
		wrapperTbl = searchWrapper_FC_RNN(paramDict, training_loader, validation_loader, outFile = outFile, verbose = True)
	elif paramDict['modelType'] == 'S2S':
		searchWrapperS2S(paramDict, training_loader, validation_loader, outFile = outFile, verbose = True)
	else:
		wrapperTbl = searchWrapperTF(paramDict, training_loader, validation_loader, outFile = outFile, verbose = True)

	print("Done!")	

	endTime = time.time()
	t_mins, t_secs = elapsedTime(startTime, endTime)
	print("Model trained and evaluated in {:02} min {:02} sec \r".format(t_mins, t_secs), end="")
	#print('Processed in %.1f seconds ' % (endTime - startTime))