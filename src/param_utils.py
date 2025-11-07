# param_utils.py
#
# Class Params definitions for parameter reading
# L. Seitsonen
#
# 14.9.2022: Code inherited from parameters.py & ftep_utils.py 
# saved with new name. /?? ttehas
#
# 15.9.2022: Added the function parseModelParameters()

import numpy as np
import pandas as pd
from collections import OrderedDict


class Params:
	''' Parameters '''
	dict = None
	list_variables = []
	
	def __init__(self, list_variables):
		self.dict = {}
		self.list_variables = list_variables
	
	def setValue(self, key, value):
		if key in self.list_variables and value != None:
			self.dict[key] = []
			self.dict[key].append(value)
		else:
			self.dict[key] = value
	
	def appendValue(self, key, value):
		if key in self.dict:
			if not key in self.list_variables:
				print('WARNING: Trying to append to non-list type variable ' + key)
				return 1
			else:
				if self.dict[key] == None:
					self.dict[key] = []
				self.dict[key].append(value)
		else:
			if key in self.list_variables and value != None:
				self.dict[key] = []
				self.dict[key].append(value)
			else:
				self.dict[key] = value
	
	def getValue(self, key):
		if key in self.dict:
			return self.dict[key]
		else:
			return None
		
	def getString(self, key):
		'''Return the value of the parameter as a string, 
		or None if the parameter does not exist or if the value is None. 
		'''
		if key in self.dict and self.dict[key] is not None:
			return str(self.dict[key])
		else:
			return None
		
	def getBoolean(self, key):
		'''Returns True if the parameter value is case-insensitive
		true or 1, False otherwise. Return False if the parameter
		value does not exist.
		'''
		if key in self.dict:
			value = str(self.dict[key]).strip()
			if value.lower() == 'true' or value == '1':
				return True

		return False

	def getInt(self, key, minvalue=None, maxvalue=None):
		'''Return the value of the parameter as an integer, 
		or None if the parameter does not exist. 
		
		Raises ValueError if the value cannot be converted to 
		an integer.
		
		Raises ValueError if the value is not within the optional 
		bounds.
		'''
		if key in self.dict:
			try:
				value = int(self.dict[key])
			except ValueError:
				raise ValueError('Invalid ' + key + ' value, should be an integer ' + str(self.dict[key]))
			
			if minvalue is not None:
				try:
					minvalue = int(minvalue)
				except:
					raise ValueError('Invalid ' + key + ' min value, should be an integer ' + str(minvalue))
			
				if value < minvalue:
					raise ValueError('Invalid ' + key + ' value, should be >= ' + str(minvalue))
			if maxvalue is not None:
				try:
					maxvalue = int(maxvalue)
				except:
					raise ValueError('Invalid ' + key + ' max value, should be an integer ' + str(maxvalue))
			
				if value > maxvalue:
					raise ValueError('Invalid ' + key + ' value, should be <= ' + str(maxvalue))
			
			return value

		else:
			return None
		
	def getFloat(self, key, minvalue=None, maxvalue=None):
		'''Return the value of the parameter as a float, 
		or None if the parameter does not exist. 
		
		Raises ValueError if the value cannot be converted to 
		a float.
		
		Raises ValueError if the value is not within the optional 
		bounds.
		'''
		if key in self.dict:
			try:
				value = float(self.dict[key])
			except ValueError:
				raise ValueError('Invalid ' + key + ' value, should be a float ' + str(self.dict[key]))
			
			if minvalue is not None:
				try:
					minvalue = float(minvalue)
				except:
					raise ValueError('Invalid ' + key + ' min value, should be a float ' + str(minvalue))
			
				if value < minvalue:
					raise ValueError('Invalid ' + key + ' value, should be >= ' + str(minvalue))
			if maxvalue is not None:
				try:
					maxvalue = float(maxvalue)
				except:
					raise ValueError('Invalid ' + key + ' max value, should be a float ' + str(maxvalue))
			
				if value > maxvalue:
					raise ValueError('Invalid ' + key + ' value, should be <= ' + str(maxvalue))
			
			return value

		else:
			return None

	def getIntList(self, key, minvalue=None, maxvalue=None, delimiter=None):
		'''Return the value of the parameter as a list of integers, 
		or None if the parameter does not exist. 
		
		Raises ValueError if the value cannot be converted to 
		an integer.
		
		Raises ValueError if the value is not within the optional 
		bounds.
		'''
		if key in self.dict:
			values = self.dict[key].split(delimiter)
			try:
				values = [ int(x) for x in values ]
			except ValueError:
				raise ValueError('Invalid ' + key + ' value, should be an integer ' + str(self.dict[key]))
			
			if minvalue is not None:
				try:
					minvalue = int(minvalue)
				except:
					raise ValueError('Invalid ' + key + ' min value, should be an integer ' + str(minvalue))
			
				for value in values:
					if value < minvalue:
						raise ValueError('Invalid ' + key + ' value, should be >= ' + str(minvalue))
			if maxvalue is not None:
				try:
					maxvalue = int(maxvalue)
				except:
					raise ValueError('Invalid ' + key + ' max value, should be an integer ' + str(maxvalue))
			
				for value in values:
					if value > maxvalue:
						raise ValueError('Invalid ' + key + ' value, should be <= ' + str(maxvalue))
			
			return values

		else:
			return None

	def getListOfIntLists(self, key, minvalue=None, maxvalue=None, delimiter=None, listdelimiter=';'):
		'''Return the value of the parameter as a list of lists of integers, 
		or None if the parameter does not exist. The lists should be separated with listdelimiter string.
		
		Raises ValueError if the value cannot be converted to 
		an integer.
		
		Raises ValueError if the value is not within the optional 
		bounds.
		'''
		if key in self.dict:
			lists = []
			list_strings = self.dict[key].split(listdelimiter)
			for list_string in list_strings:
				values = list_string.split(delimiter)
				try:
					values = [ int(x) for x in values ]
				except ValueError:
					raise ValueError('Invalid ' + key + ' value, should be an integer ' + str(self.dict[key]))
				
				if minvalue is not None:
					try:
						minvalue = int(minvalue)
					except:
						raise ValueError('Invalid ' + key + ' min value, should be an integer ' + str(minvalue))
				
					for value in values:
						if value < minvalue:
							raise ValueError('Invalid ' + key + ' value, should be >= ' + str(minvalue))
				if maxvalue is not None:
					try:
						maxvalue = int(maxvalue)
					except:
						raise ValueError('Invalid ' + key + ' max value, should be an integer ' + str(maxvalue))
				
					for value in values:
						if value > maxvalue:
							raise ValueError('Invalid ' + key + ' value, should be <= ' + str(maxvalue))
				
				lists.append(values)
			
			return lists

		else:
			return None

	def getFloatList(self, key, minvalue=None, maxvalue=None, delimiter=None):
		'''Return the value of the parameter as a list of floats, 
		or None if the parameter does not exist. 
		
		Raises ValueError if the value cannot be converted to 
		a float.
		
		Raises ValueError if the value is not within the optional 
		bounds.
		'''
		if key in self.dict:
			values = self.dict[key].split(delimiter)
			try:
				values = [ float(x) for x in values ]
			except ValueError:
				raise ValueError('Invalid ' + key + ' value, should be a float ' + str(self.dict[key]))
			
			if minvalue is not None:
				try:
					minvalue = float(minvalue)
				except:
					raise ValueError('Invalid ' + key + ' min value, should be a float ' + str(minvalue))
			
				for value in values:
					if value < minvalue:
						raise ValueError('Invalid ' + key + ' value, should be >= ' + str(minvalue))
			if maxvalue is not None:
				try:
					maxvalue = float(maxvalue)
				except:
					raise ValueError('Invalid ' + key + ' max value, should be a float ' + str(maxvalue))
			
				for value in values:
					if value > maxvalue:
						raise ValueError('Invalid ' + key + ' value, should be <= ' + str(maxvalue))
			
			return values

		else:
			return None
		
	def __repr__(self):
		s = '';
		for (k, v) in self.dict.items():
			if s != '':
				s = s + ', '
			s = s + k + '=' + str(v)
		return s


	def readFile(self, file):
		'''Reads a properties file with key=value pairs

		If value starts with def (case insensitive) does not
		modify the default value. If value equals none (case
		insensitive) sets the value explicitly to None.
		Lines starting with # are comments.
		Leading and trailing whitespace is removed.
		'''
		separator = '='

		with open(file) as f:

			for line in f:
				if line.startswith('#'):
					continue

				if separator in line:

					# Find the name and value by splitting the string
					name, value = line.split(separator, 1)
					# strip() removes white space from the ends of strings
					name = name.strip()
					value = value.strip()
					if value.startswith('"') and value.endswith('"') and len(value) >= 2:
						value = value[1:-1]
					if len(value) > 0:
						if value.lower() == 'none':
							# Explicitly set parameter to None, overrides defaults
							self.setValue(name, None)
						elif not value.lower().startswith('def'):
							# def is used to mark default value, ignore such lines
							self.setValue(name, value)


# parseModelParams(...)
#
# An improved version that reads the parameter type file name (parameterTypeFile)
# from the parameter file itself.
#
# A function to read parameters from textfile using the methods of Params class.
# Inputs:
#       parameterFile:      text file with parameters as (name, value) pairs ('=' as separator).
# Returns:
#       paramDict:          An ordered dictionary of the read parameters
#       params.dict:        A dictionary of the parameters (as strings) as returned by
#                           the method params.readFile()
#       paramList:          A list of the parameters (read from the parameterTypeFile) 

def parseModelParams(parameterFile, nullReturn = None, verbose = True):

	# Read parameters for modelling:
	list_variables = []
	params = Params(list_variables)
	if verbose:
		print("parameterFile = ", parameterFile)
	params.readFile(parameterFile)
	
	# Read parameter type file name from the parameter file:
	try:
		parameterTypeFile = params.getString('parameterTypeFile')

		# Read the parameter types into a pandas dataframe.
		# NOTE: the parameter type file contains type definitions ('int', float', 
		# 'string', 'intList', 'floatList', 'stringList' and 'boolean') for the
		# parameter reading. Separaten parameter type files for DNN & RF models:
		parameterType_df = pd.read_csv(parameterTypeFile, delimiter='\t', index_col=False)
	except:
		# Set default (local) location of parameter typefile:
		parameterTypeFile ='C:\\PROJECTS\\2023_ARTISDIG\\WP4\\AI_EMULATOR\\ART_REPO\\seq2seq_parameterTypes.txt'
	
		parameterType_df = pd.read_csv(parameterTypeFile, delimiter='\t', index_col=False)

	# Create a parameter name & type lists from the pd dataframe:
	paramList = parameterType_df['Name'].tolist()
	paramTypes = parameterType_df['Type'].tolist()

	## Create a parameter list from the input parameters:
	##paramList = list(params.dict.keys())

	# Loop through all parameters and use class 'Params' methods
	# for reading according to each parameter's type:

	paramDict = OrderedDict()
	#paramDict = {}

	for i, thisParam in enumerate (paramList):
		#print("thisParam = ", thisParam)
		#print("paramTypes[i] = ", paramTypes[i])
		
		try:
			if paramTypes[i] == 'int':
				paramDict[thisParam] = params.getInt(thisParam)
			
			if paramTypes[i] == 'intList':
				paramDict[thisParam] = params.getIntList(thisParam)
			
			if paramTypes[i] == 'float':
				paramDict[thisParam] = params.getFloat(thisParam)
		
			if paramTypes[i] == 'floatList':
				paramDict[thisParam] = params.getFloatList(thisParam)
		
			if paramTypes[i] == 'string':
				paramDict[thisParam] = params.getString(thisParam)
			
			if paramTypes[i] == 'stringList':
				paramStr = params.getValue(thisParam)
				if paramStr is not None:
					paramDict[thisParam] = paramStr.split()
				else:
					if nullReturn == None:
						paramDict[thisParam] = None
					if nullReturn == 'empty':
						paramDict[thisParam] = []
					
			if paramTypes[i] == 'boolean':
				paramDict[thisParam] = params.getBoolean(thisParam)
		except:
			raise ValueError('Problem reading parameter ' + thisParam)
			
	if verbose:
		print("\nParameters from: " + parameterFile +"\n")
		for key, value in paramDict.items():
			print(key + ': ', value)
		
	return paramDict, params.dict, paramList
        

# parseModelParameters(...)
#
# A function to read parameters from textfile using the methods of Params class.
# Inputs:
#       parameterFile:      text file with parameters as (name, value) pairs ('=' as separator).
#       parameterTypeFile:  a separate text file (tab separated) with type for each
#                           parameter in parameterFile
# Returns:
#       paramDict:          An ordered dictionary of the read parameters
#       params.dict:        A dictionary of the parameters (as strings) as returned by
#                           the method params.readFile()
#       paramList:          A list of the parameters (read from the parameterTypeFile) 

def parseModelParameters(parameterFile, parameterTypeFile, nullReturn = None, verbose = True):
	# Read the parameter types into a pandas dataframe.
	# NOTE: the parameter type file contains type definitions ('int', float', 
	# 'string', 'intList', 'floatList', 'stringList' and 'boolean') for the
	# parameter reading. Separaten parameter type files for DNN & RF models:
	parameterType_df = pd.read_csv(parameterTypeFile, delimiter='\t', index_col=False)

	# Create a parameter name & type lists from the pd dataframe:
	paramList = parameterType_df['Name'].tolist()
	paramTypes = parameterType_df['Type'].tolist()

	'''
	# Convert the parameter type data frame to dictionary containing keys 'index', 'columns' and 'data':
	paramTypesRaw = parameterType_df.to_dict('split')
	#print("paramTypesRaw = \n", paramTypesRaw)

	# Further convert the data behind key'data' (now as two-column list) to dictionary
	# containing the parameter names as keys, and parameter types as values:
	paramTypes = dict(paramTypesRaw['data'])
	#print("paramTypes = \n", paramTypes)
	'''

	# Read parameters for modelling:
	list_variables = []
	params = Params(list_variables)
	params.readFile(parameterFile)

	## Create a parameter list from the input parameters:
	##paramList = list(params.dict.keys())

	# Loop through all parameters and use class 'Params' methods
	# for reading according to each parameter's type:

	paramDict = OrderedDict()
	#paramDict = {}

	for i, thisParam in enumerate (paramList):
		#print("thisParam = ", thisParam)
		#print("paramTypes[i] = ", paramTypes[i])
		
		try:
			if paramTypes[i] == 'int':
				paramDict[thisParam] = params.getInt(thisParam)
			
			if paramTypes[i] == 'intList':
				paramDict[thisParam] = params.getIntList(thisParam)
			
			if paramTypes[i] == 'float':
				paramDict[thisParam] = params.getFloat(thisParam)
		
			if paramTypes[i] == 'floatList':
				paramDict[thisParam] = params.getFloatList(thisParam)
		
			if paramTypes[i] == 'string':
				paramDict[thisParam] = params.getString(thisParam)
			
			if paramTypes[i] == 'stringList':
				paramStr = params.getValue(thisParam)
				if paramStr is not None:
					paramDict[thisParam] = paramStr.split()
				else:
					if nullReturn == None:
						paramDict[thisParam] = None
					if nullReturn == 'empty':
						paramDict[thisParam] = []
					
			if paramTypes[i] == 'boolean':
				paramDict[thisParam] = params.getBoolean(thisParam)
		except:
			raise ValueError('Problem reading parameter ' + thisParam)
			
	if verbose:
		print("\nParameters from: " + parameterFile +"\n")
		for key, value in paramDict.items():
			print(key + ': ', value)
		
	return paramDict, params.dict, paramList