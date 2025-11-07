# artisTrain.py.py
#
# This is a stand-alone version of the code originally written in Seq2seq_model.ipynb
#
# The purpose is to be able to train + verify a PyTorch model from command line
# interface and in Linux operation system (in CSC machines).

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
from pytorchtools import trainingLoop, elapsedTime
from pytorchtools import evaluate, trainValidPredictions

print("")
print("Current Python Version -", python_version())
print("Pandas version -", pd.__version__)


if __name__ == '__main__':

	startTime = time.time()

	# --------------------------------------------------------------------------------------------
	# Argument handling
	# --------------------------------------------------------------------------------------------
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
	# Model definition
	# ------------------------------------------------------------
	model, criterion, optimizer, scheduler = initModel(paramDict)	

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
	# Model training
	# ------------------------------------------------------------
	print("3. Train the model ...")
	model, trainvalidLosses, bestValidLoss, monTrainLoss = trainingLoop(model, paramDict, training_loader, validation_loader, optimizer, criterion, verbose = False)

	# ---------------------------------------------------------------------------
	# Compute test set response and de-normalize the results
	# ---------------------------------------------------------------------------
	# Define loss % time monitoring step (= nbr of batches):
	print("4. Compute test set response & de-normalize ...")
	
	monStep = 50

	evalParams = {
		'modelType': paramDict['modelType'],
		'loss_function': paramDict['loss_function'],
		'phase': 'test',
		'monStep': monStep,
		'verbose': False
		}
	#start_time = time.time()
	foo, targetsTest, predsTest = evaluate(model, test_loader, criterion, evalParams)

	tgtStatsFile = os.path.join(paramDict['outPath'], 'targetStats.txt')
	targetsTest_denorm, predsTest_denorm = predsdeNorm(targetsTest, predsTest, statsFile = tgtStatsFile)

	# ---------------------------------------------------------------------------
	# Compute figures of merit (RMSE%, BIAS% & R2 by diffrent views of data)
	# gen_utils_art.py/computeResults()
	# ---------------------------------------------------------------------------
	print("5. Compute FoMs ...")

	# Extract the species presence/absence table from:
	exstFile = os.path.join(paramDict['outPath'],'testSetData.csv')

	fomsFlat, fomsPerYear, fomsPerCase, fomsFlatTbl, fomsPerYearDict, fomsPerCaseDict = computeResults(targetsTest_denorm, predsTest_denorm, paramDict['targetVars'], exstFile = exstFile, outPath = paramDict['outPath'])

	# ---------------------------------------------------------------------------
	# Join the results (figures of merits, predictions, and optionally targets)
	# with the input test data set: 
	# ---------------------------------------------------------------------------
	print("3. Join results and test data set + save ...")

	testDataFile = os.path.join(paramDict['outPath'], 'testSetData.csv')
	resultsJoined_df = joinTestDataWithResults(paramDict, testDataFile, predsTest_denorm, fomsPerCaseDict, 
											   targets = targetsTest_denorm)

	# ---------------------------------------------------------------------------
	# Produce predictions for training & validation data sets + save with inputs
	# ---------------------------------------------------------------------------
	if paramDict['cascadeTgtVars'] is not None:
		print("4. Produce predictions for training & validation data + save ...")
		trainValidPredictions(paramDict, model, criterion, monStep = 20)
	
	print("Done!")	

	endTime = time.time()
	t_mins, t_secs = elapsedTime(startTime, endTime)
	print("Model trained and evaluated in {:02} min {:02} sec \r".format(t_mins, t_secs), end="")
