# This file contains tools for reading and manipulating forest variable data
# provided by Finnish Forest Centre (Suomen Metsäkeskus, SMK)
#
# https://www.metsakeskus.fi/fi/avoin-metsa-ja-luontotieto/aineistot-paikkatieto-ohjelmille/paikkatietoaineistot
#
#
K_DISK_PRESENT = True

import os
import fnmatch
import numpy as np
import pandas as pd
import geopandas as gpd
from os import listdir
from os.path import isfile, join, splitext
import matplotlib.pyplot as plt

if K_DISK_PRESENT:
	from py4R_tools import recurse_r_tree
	from rpy2 import robjects


from collections import OrderedDict


# read_geopackages(inputFolder, sampleplot_fields = None, stratum_fields = None)
#
# This function reads the tables 'sampleplot' and 'stratum' from geopackage files (ext = '.gpkg')
# contained in the directory defined by 'inputFolder', and concatenates the data into two result
# GeoDataFrames. The fields read from the tables may be explicitly specified by the parameters 
# 'sampleplot_fields' and 'stratum_fields' (default = None = all fields read). If desired,
# the two tabled may be merged into a single output GeoDataFrame by the field 'sampleplotid'.
# In this case the first output geoDataFrame contains the merged data. The second output dataframe
# contains the stratum data.
#
# Parameters:
#
# inputFolder		(string; path) Path to the inputFolder containing the geopackage files
# sampleplot_fields	(list of strings) The fields to be read from the table 'sampleplot' 
#					(default = None = read all fields)
# stratum_fields	(list of strings) The fields to be read from the table 'stratum' 
#					(default = None = read all fields)
# mergeTables		(bool) A flag indicating, if the two tables are to be merged (outer 
#					join) according to field 'sampleplotid'
# addAreaID			(bool) A flag indicating if an area ID code is to be added to the 
#					output geo-dataframe. The areaID codes corresponds to the text
#					strings given in list variable 'areaIDstrs'. The area
#					ID assigned to the output geo-dataframe is the relative position
#					(index) the corresponding string in the list 'areaIDstrs', i.e. 
#					if areaIDstrs = ['EKa', 'EPo', 'EtSa', 'KaHa'], then the data from
#					the geopackage file containing the string 'EKa' in filename gets
#					the area ID number 0, and the geopackage file with string 'EtSa'
#					the ID number 2. If none of the strings in 'areaIDstrs' are included
#					in the geopackage file name, then the ID number 999 will be assigned.
#
#
# SMK Inventory plot data (accessed 24.4.2023):
#
# https://aineistot.metsaan.fi/avoinmetsatieto/Inventointikoealat/Maakunta/
#
# Geopackage Documentation (accessed 24.4.2023):
#
# https://www.metsakeskus.fi/sites/default/files/document/tietotuotekuvaus-inventointikoealat.pdf<br>
# https://www.metsakeskus.fi/sites/default/files/document/inventointikoealat-tietokantakaavio.pdf

# NOTE: Optionally add an identifier number column to the output geo-dataframe 
# separating the data origin (forset district). So far only the identifiers in 'areaIDstrs' have been
# defined. The corresponding string must be found in the input geopackage filename:
areaIDstrs = ['EKa', 'EPo', 'EtSa', 'KaHa', 'Kai', 'KePo', 'KeSu', 'KyLa', 'Lappi_E', 'Lappi_P', 'PaHa', 'PiMa', 'PoKa', 'PoMa', 'PoPo', 'PoSa', 'SaKu', 'UuMa', 'VaSu']


def read_geopackages(inputFolder, sampleplot_fields = None, stratum_fields = None, mergeTables = False, addAreaID = False, verbose = False):
    
	# Gather list of geopackage files from the inputFolder given as function input:
	files = [os.path.join(inputFolder, f) for f in listdir(inputFolder) if (isfile(join(inputFolder, f)) and splitext(f)[1]=='.gpkg' )]

	for thisFile in files:
		if verbose:
			print("processing : ", thisFile)
			
		# Use geopandas to open the database tables of the used geopackage
		
		# Seems that some of the geopackages do not have layer "stratum":
		try:
			this_stratum_gdf = gpd.read_file(thisFile, layer="stratum", ignore_geometry=False)
		except:
			print("layer= 'stratum' not found" )
			this_stratum_gdf = None
			
		try:
			this_sampleplot_gdf = gpd.read_file(thisFile, layer="sampleplot", ignore_geometry=False)
		except:
			print("layer= 'sampleplot' not found" )
			this_sampleplot_gdf = None
		
		# Select only required columns
		if stratum_fields is not None:
			this_stratum_gdf = this_stratum_gdf[stratum_fields]
			
		if sampleplot_fields is not None:
			this_sampleplot_gdf = this_sampleplot_gdf[sampleplot_fields]
		
		# Add the area ID code if desired:
		if addAreaID:
			areaID = [ii for ii, x in enumerate(areaIDstrs) if x in thisFile]
			if areaID != []:
				areaIDnbr = areaID[0]
			else:
				areaIDnbr = 999
			
			this_sampleplot_gdf['areaID'] = np.repeat(areaIDnbr, this_sampleplot_gdf.shape[0])
			
		if verbose:
			print("this_sampleplot_gdf.shape = ", this_sampleplot_gdf.shape)
			print("this_stratum_gdf.shape = ", this_stratum_gdf.shape)
		
		if thisFile == files[0]:
			# Generate output data frames, if this was the first geopackage in the list:
			stratum_gdf = this_stratum_gdf
			sampleplot_gdf = this_sampleplot_gdf
		else:
			# Append to the final dataframes (column order is important!)
			stratum_gdf = stratum_gdf.append(this_stratum_gdf)
			sampleplot_gdf = sampleplot_gdf.append(this_sampleplot_gdf)
			
	if verbose:
		print("sampleplot_gdf.crs = ", sampleplot_gdf.crs)
		print("merge: stratum_gdf.shape = ", stratum_gdf.shape)
		print("merge: stratum_gdf.shape = ", stratum_gdf.shape)

	if mergeTables:
		if sampleplot_gdf is not None:
			# Save the crs of 'sampleplot_gdf' (see below):
			sampleplot_gdf_crs = sampleplot_gdf.crs
			merge_df = sampleplot_gdf.merge(stratum_gdf, how='outer', on='sampleplotid')
			
			# The .merge() method returns an ordinary dataframe ´(even though the documentation
			# suggests otherwise). Thus convert the dataframe back to geoDataFrame:
			sampleplot_gdf = gpd.GeoDataFrame(merge_df, geometry=gpd.points_from_xy(merge_df['xcoordinate'].values, 
				merge_df['ycoordinate'].values, z=merge_df['zcoordinate'].values, crs=sampleplot_gdf_crs))
		
	if verbose:
		print("sampleplot_gdf.crs = ", sampleplot_gdf.crs)
		print("merge: sampleplot_gdf.shape = ", sampleplot_gdf.shape)
		
	return sampleplot_gdf, stratum_gdf
	

# filterDataFrame(df, filters)
# This function may be used to filter out data rows (cases) from a DataFrame (or GeoDataFrame).
# Parameters:
#
# df		= (DataFrame or GeoDataFrame) The input data frame to be filtered.
# filters	= (list of strings or string) The filter rules provided as list of strings,
#				or as a single string with individual filter rules separated with semicolon(;).
# subSet	= (an array of discrete values) If a subset of the input geodataframe
#				are to be selected based on (a large amount) of discrete values of
#				some column, th user may pass the values through this variable.
#				In this case the desired data rows are selected by the rule:
#				'<column_var_name> in @subSet', which rule must be included in 'filters'.
#				Note: The local variable 'subSet' is not referred within the function
#				code at all, but it will (and must) be included in the filter string!
#				(see: Example 3., and the link to Pandas documentation for details)
#				Note 2: The function accepts only once the filter rule with the passed subSet.
#				If more than one subsets must be filtered, the the function must be called
# 				several times. For other types of filters the number of filters is unlimited.
#
# For the rules, see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html
# or the examples below.
#
# Examples:
#
# 1. Filter a dataframe 'df' that contains the columns 'PLOT_V' and 'DOY' the  to cases for
# which PLOT_V is in the range 0 <= PLOT_V < 600. Also accept only data starting from
# DOY = 121 (= 1st of May):
#
# filterDataFrame(df, ['PLOT_V >= 0', 'PLOT_V < 600', 'DOY > 120']), or
# filterDataFrame(df, 'PLOT_V >= 0; PLOT_V < 600; DOY > 120')
#
# 2. Select forest variable data for which 'meanheight' is above 1.5 m, and 'treespecies' 
# is one of the defined values (the last rule defines an 'or' type rule).
#
# filterDataFrame(df, ['meanheight > 1.5', 'treespecies in [3, 4]']), or
# filterDataFrame(df, 'meanheight > 1.5; treespecies in [3, 4]')
#
# 3. Use of 'subSet' filter:
# treeSpeciesSet = [1, 2, 3, 4]
# filterDataFrame(df, ['meanheight > 1.5', 'treespecies in @subSet'], subSet = treeSpeciesSet)
	
def filterDataFrame(df, filters, subSet = None):

	try:
		# Check if filters are given as a string:
		if isinstance(filters, str):
			filters = filters.split(';')

		for thisFilter in filters:
			df = df.query(thisFilter)
			
	except TypeError as err:
		print('filterDataFrame(): Wrong filter format:', err)
	except ValueError as err:
		print('filterDataFrame(): Invalid filter string:', err)
	except KeyError as err:
		print('filterDataFrame(): Invalid filter data:', err)
	except Exception as err:
		print('filterDataFrame(): Unknown Error:', err)
		
	return df


# replace_column_names(df, hdrs_old, hdrs_new)
# Replace the (selected) column names of a Pandas DataFrame with user-defined names.
#
# Parameters:
# df (pandas.DataFrame): The DataFrame whose column names needs to be replaced.
# hdrs_old (str): The old column names to be replaced.
# hdrs_new (str): The new column names to be assigned.
#
# Returns:
# pandas.DataFrame: The DataFrame with updated column name.

def replace_column_names(df, hdrs_old, hdrs_new):
    if len(hdrs_new) != len(hdrs_old):
        raise ValueError("Number of new and old column names does not match.")
		
    new_columns = dict(zip(hdrs_old, hdrs_new))
    df = df.rename(columns=new_columns)
    
    return df

# forVarData_gpkg2prebas()
#
# Organize the forest variable data into the format of table 
# ..\DOC\initVarIn.txt (or: ..\DOC\initVarIn.xlsx):
#
# Parameters:
#
# gdf_in 			= the input geoDataFrame (read from Metsäkeskus' 
#					geopackage files (ext = '.gpkg'))
# filterStr			= filter rule string (or a list of strings) for
#					selecting the desired rows of the input data frame
#					(see filterDataFrame() above)
# speciesSubSet		= (list of int) The subset of tree species to be selected
#					(filtered) from the input geoDataFrame. 
#					NOTE: To include more broadleaved species into 'broadleaved' 
#					category, additional aggregating code must be added to this
#					function, or to subsequent processing stages.
# forVarHdrs_in		= The headers of the input geoDataFrame columns to selected
# forVarHdrs_out	= The headers of the output geoDataFrame. These replace the 
#					headers in 'forVarHdrs_in'. The length of 'forVarHdrs_out'
#					must match the lenght of 'forVarHdrs_in'.

def fieldPlot2prebas(gdf_in, 
	filterStr = 'treespecies in @subSet',
	speciesSubSet = [1, 2, 3], 
	forVarHdrs_in = ['sampleplotid', 'stratumnumber', 'treespecies', 'meanage', 'meanheight', 'meandiameter', 'basalarea'], 
	forVarHdrs_out = ['site', 'Layers', 'speciesID', 'age', 'H', 'D', 'BA']):

	# Filter the input geoDataFrame (default: the data for the species 1 = pine, 2 = spruce, 
	# and 3 = downy birch (betula pendula) only):
	# NOTE: If additional species (e.g. all broadleaved species) are to be included in the 
	#'broadleved' category, then the specific code must be added 
	# (forestVarData_tools.py/aggregateSpeciesData()).
	gdf_in_filt = filterDataFrame(gdf_in, filterStr, subSet=speciesSubSet)

	# Select only the desired columns of the sample plot data geodataframe:
	forVarData_gdf = gdf_in_filt[forVarHdrs_in]

	# Replace the input geoDataPackege column names with the ones expected by MultiPrebas:
	forVarData_gdf = replace_column_names(forVarData_gdf, forVarHdrs_in, forVarHdrs_out)

	# The next row produces a warning message:
	#forVarData_gdf[['Hc', 'A']] = np.zeros((forVarData_gdf.shape[0], 2))
	forVarData_gdf[['Hc', 'A']] = (np.nan, np.nan)
	#forVarData_gdf.head()

	# Extract unique site ID's. This will be used later when generating
	# the input table 'siteInfo' for prebasMultiSiteWrapper.r:
	# NOTE that the Pandas function pd.unique() returns the unique data set 
	# in order of appearance in the original data set:
	siteIDs_uniq = pd.unique(forVarData_gdf['site'])

	return forVarData_gdf, gdf_in_filt, siteIDs_uniq


# siteInfo2prebas()
#
# This function produces the siteInfo data table for Prebas/multiPrebar programs.
# The function may be used also in "randomize" mode to generate augmented data (for
# Prebas) when generating data sets for the AI emulator training. In this case the 
# user must supply a non-zero 'noisePerc' variable, and instead of providing 
# a file name for 'climID's' (with climID's sampled from real site locations)  
# provide an integer (C) for generating random climID's 1-C.
#
# sampleplot_gdf_filt	(Pandas geodataframe) A geodataframe with forestry data
#						(obtained from FFC geodatapackages) filtered properly
#						by fieldPlot2prebas()
# climIDs				(string or int) 1. string: file name of a *.csv text file 
#						containing climID's sampled from the site loactions
#						2. integer: An integer (C) indicating to generate random 
#						climID's 1 - C for the sites
# noisePerc				(float) If non-zero, indicates that the siteType variable and
#						the default values of pre-defined siteInfo variables shall
#						be added with gaussian random noise equal to the percentage
#						of the default value.

# sites_uniq			(Pandas dataframe) A list of unique site ID numbers (removed 3.11.2023 ??/ttehas)

def siteInfo2prebas(sampleplot_gdf_filt, climIDs = None, noisePerc = 0.0, verbose = False):

	sites_uniq = pd.unique(sampleplot_gdf_filt['sampleplotid'])
	nSites = sites_uniq.shape[0]
	
	# ===============================================================================
	# 1. Generate siteInfo dataframe using (mostly) default values:
	# -------------------------------------------------------------------------------
	# Define siteInfo headers (hard coded). Note that 'climID' is still missing:
	siteInfoColHdrs = ['siteID', 'siteType', 'SWinit', 'CWinit', 'SOGinit', 
						'Sinit', 'nLayers', 'nSpecies', 'soildepth', 'effective field capacity', 'permanent wilting point']

	# 'siteInfo' default vector ('siteID',  and 'siteType' are None), (hard coded):
	siteInfo = pd.DataFrame([[None,None,160,0,0,20,3,3,413.,0.45,0.118]], columns = siteInfoColHdrs)

	# Repeat the default vector 'nSites' times:
	siteInfo = pd.concat([siteInfo]*nSites, ignore_index=True)

	if noisePerc > 0:
		# If the given randomizin percentage is greater than zero (= default), then
		# add random Gaussian noise to the defaults siteInfo variables:
		# ['SWinit', 'CWinit', 'SOGinit', 'Sinit', 'soildepth', 'effective field capacity', 'permanent wilting point']
		siteInfo = addRndNoise2siteInfo(siteInfo, noisePerc = noisePerc, CWinit_std = 0.05, SOGinit_std = 0.05, ignoreCols = ['nLayers', 'nSpecies'], verbose = verbose)
	
	# Assign the unique 'siteID' (from 'forVarData' above) to siteInfo 
	# (other data identical for all rows):
	siteInfo['siteID'] = sites_uniq
	if verbose:
		print("siteInfo.shape = ", siteInfo.shape)

	# ===============================================================================
	# 2. Join the fertility class / site type from the 'sampleplot_gdf_filt' dataframe
	# -------------------------------------------------------------------------------
	# containing filtered forest inventory data:
	# Extract site types from the corresponding locations in 'sampleplot_gdf_filt':
	siteTypes2merge = sampleplot_gdf_filt[['sampleplotid', 'fertilityclass']].drop_duplicates()

	# Replace headers (Note: different headers used for site ID# in forest variable array and site info array):
	siteTypes2merge = replace_column_names(siteTypes2merge, ['sampleplotid', 'fertilityclass'], ['siteID', 'siteType'])

	# Ensure that the site types are joined to correct sites (join on 'siteID') by
	# constructing an intermediate variable 'siteInfo_merge':
	siteInfo_merge = siteInfo[['siteID']].merge(siteTypes2merge, how='inner', on='siteID')

	# Replace the None's in the 'siteInfo' table:
	siteInfo['siteType'] = siteInfo_merge['siteType']
	
	if noisePerc > 0:
		# If the given randomizin percentage is greater than zero (= default), then
		# change the siteType variable randomly by one class (or level), i.e. add
		# randomly +/- 1 to the real siteType. Ensure that the siteType range does
		# not change!
		val_min = min(siteInfo['siteType'].values)
		val_max = max(siteInfo['siteType'].values)

		siteInfo['siteType'] = siteInfo['siteType'].apply(add_random_values, args=(val_min, val_max))

	# ===============================================================================
	# 3. Join the 'climID' for each plot location from file:
	# -------------------------------------------------------------------------------
	# Read 'climID' for each site from *.csv file (climID's sampled from climID image):
	if isinstance(climIDs, str):
		climID_df = pd.read_csv(climIDs)
		# Drop duplicate rows (this is due to a bug when creating AI emulator test set #1
		# and should be unnecessary later):
		climID_df.drop_duplicates(inplace = True)
		
		# Merge climID's from climID_df:
		siteInfo = siteInfo.merge(climID_df[['siteID', 'climID']], how='inner', on='siteID')
	else:
		# Assign random 'climIDs' (NOTE: all climIDs 1 - nClimIDs must be present in the table):
		nClimIDs = climIDs
		siteInfo['climID'] = np.ceil(nClimIDs*np.random.rand(nSites,1))
		
	# Reorder columns to match the order expected by multiPrebas.r:
	siteInfoColHdrs = ['siteID', 'climID', 'siteType', 'SWinit', 'CWinit', 'SOGinit', 
						'Sinit', 'nLayers', 'nSpecies', 'soildepth', 'effective field capacity', 'permanent wilting point']
	siteInfo = siteInfo[siteInfoColHdrs]

	if verbose:
		print("nSites = ", nSites)
		print("sampleplot_gdf_filt.shape = ", sampleplot_gdf_filt.shape)
		print("siteInfo_merge.shape = ", siteInfo_merge.shape)
		print("climID_df.columns = ", climID_df.columns)
		print("climID_df.shape = ", climID_df.shape)
		print("siteInfo.shape = ", siteInfo.shape)

	return siteInfo, sites_uniq


# addRndNoise2siteInfo()
#
# This function adds random noise to certain variables of (default) siteInfo
# dataframe.
#
# The range of the random noise is the 'noisePerc' percenatge taken 
# from the default value of each parameter, e.g. with SWinit the noise is:
# noise_SWinit = np.random.randn(nSites,1) * 160 * noisePerc/100
# and the obtained randomized values for SWinit = 160 + noise_SWinit:
# NOTE: The variables 'CWinit', 'SOGinit' to be treated differently!
#
# siteInfo_default	(dataframe) The siteInfo dataframe (with defaultvalues)
#
# noisePerc			(float) The percentage of noise wrt. to the default value
#					to be added to the default values of ['SWinit', 'Sinit',
#					'soildepth', 'effective field capacity', 'permanent wilting point']
#					The defaults for parameters ['CWinit', 'SOGinit'] are zero,
#					so an absolute level (stdev) of noise will be specified with
#					parameters 'CWinit_std' and 'SOGinit_std'.
#
# 'CWinit_std'		(float) The absolute noise level (stdev) for 'CWinit'. The
#					absolute values of the noise will be produced.
#					
# 'SOGinit_std'		(float) The absolute noise level (stdev) for 'SOGinit'. The
#					absolute values of the noise will be produced.
#
# 'ignoreCols'		(list of strings) The columns to innore ('nLayers' & 'nSpecies').

def addRndNoise2siteInfo(siteInfo_default, noisePerc = 2.0, CWinit_std = 0.05, SOGinit_std = 0.05, ignoreCols = ['nLayers', 'nSpecies'], verbose = False):

	# Calculate the standard deviation for the noise
	std_dev = (noisePerc / 100) * siteInfo_default
	std_dev['CWinit'].replace(to_replace=0, value=CWinit_std, inplace=True)
	std_dev['SOGinit'].replace(to_replace=0, value=SOGinit_std, inplace=True)

	# reset std values to zero for ignoreCols:
	std_dev[ignoreCols] = 0.0

	if verbose:
		print("std_dev = ", std_dev)

	# Generate random Gaussian noise with the same shape as the DataFrame
	noise = np.abs(np.random.normal(0, std_dev, size=siteInfo_default.shape))

	# Add the noise to the DataFrame
	output_df = siteInfo_default + noise

	# Change the NaN's to none to keep the original format:
	output_df = output_df.where(pd.notnull(output_df), None)

	# Display the noisy DataFrame
	if verbose:
		print("output_df = ", output_df)

	return output_df


# add_random_values()
#
# Function to add random integers from range add_values (default = [-1, 0, 1])
# to the column values while constraining within val_min and val_max.
# see usage in fuction: siteInfo2prebas)

def add_random_values(value, val_min, val_max, add_values = [-1, 0, 1]):

    new_value = value + np.random.choice(add_values)
    return min(max(new_value, val_min), val_max)



def siteInfo2prebas_orig(sampleplot_gdf_filt, sites_uniq, climIDs = None, verbose = False):

	nSites = sites_uniq.shape[0]
	
	# ===============================================================================
	# 1. Generate siteInfo dataframe using (mostly) default values:
	# -------------------------------------------------------------------------------
	# Define siteInfo headers (hard coded). Note that 'climID' is still missing:
	siteInfoColHdrs = ['siteID', 'siteType', 'SWinit', 'CWinit', 'SOGinit', 
						'Sinit', 'nLayers', 'nSpecies', 'soildepth', 'effective field capacity', 'permanent wilting point']

	# 'siteInfo' default vector ('siteID',  and 'siteType' are None), (hard coded):
	siteInfo = pd.DataFrame([[None,None,160,0,0,20,3,3,413.,0.45,0.118]], columns = siteInfoColHdrs)

	# Repeat the default vector 'nSites' times:
	siteInfo = pd.concat([siteInfo]*nSites, ignore_index=True)

	# Assign the unique 'siteID' (from 'forVarData' above) to siteInfo 
	# (other data identical for all rows):
	siteInfo['siteID'] = sites_uniq
	if verbose:
		print("siteInfo.shape = ", siteInfo.shape)

	# ===============================================================================
	# 2. Join the fertility class / site type from the 'sampleplot_gdf_filt' dataframe
	# -------------------------------------------------------------------------------
	# containing filtered forest inventory data:
	# Extract site types from the corresponding locations in 'sampleplot_gdf_filt':
	siteTypes2merge = sampleplot_gdf_filt[['sampleplotid', 'fertilityclass']].drop_duplicates()

	# Replace headers (Note: different headers used for site ID# in forest variable array and site info array):
	siteTypes2merge = replace_column_names(siteTypes2merge, ['sampleplotid', 'fertilityclass'], ['siteID', 'siteType'])

	# Ensure that the site types are joined to correct sites (join on 'siteID') by
	# constructing an intermediate variable 'siteInfo_merge':
	siteInfo_merge = siteInfo[['siteID']].merge(siteTypes2merge, how='inner', on='siteID')

	# Replace the None's in the 'siteInfo' table:
	siteInfo['siteType'] = siteInfo_merge['siteType']

	# ===============================================================================
	# 3. Join the 'climID' for each plot location from file:
	# -------------------------------------------------------------------------------
	# Read 'climID' for each site from *.csv file (climID's sampled from climID image):
	if isinstance(climIDs, str):
		climID_df = pd.read_csv(climIDs)
		# Drop duplicate rows (this is due to a bug when creating AI emulator test set #1
		# and should be unnecessary later):
		climID_df.drop_duplicates(inplace = True)
		
		# Merge climID's from climID_df:
		siteInfo = siteInfo.merge(climID_df[['siteID', 'climID']], how='inner', on='siteID')
	else:
		# Assign random 'climIDs' (NOTE: all climIDs 1 - N must be present in the table).
		# This is only for testing purposes:
		nClimIDs = climIDs
		siteInfo['climID'] = np.ceil(nClimIDs*np.random.rand(nSites,1))
		
	# Reorder columns to match the order expected by multiPrebas.r:
	siteInfoColHdrs = ['siteID', 'climID', 'siteType', 'SWinit', 'CWinit', 'SOGinit', 
						'Sinit', 'nLayers', 'nSpecies', 'soildepth', 'effective field capacity', 'permanent wilting point']
	siteInfo = siteInfo[siteInfoColHdrs]

	if verbose:
		print("nSites = ", nSites)
		print("sampleplot_gdf_filt.shape = ", sampleplot_gdf_filt.shape)
		print("siteInfo_merge.shape = ", siteInfo_merge.shape)
		print("climID_df.columns = ", climID_df.columns)
		print("climID_df.shape = ", climID_df.shape)
		print("siteInfo.shape = ", siteInfo.shape)

	return siteInfo



#def buildDataSet4AI(forVarDataFile, siteInfoFile, climDataFile, speciesStr = ['pine', 'spr', 'bl'], verbose = False):
def buildDataSet4AI_v01(forVarDataFile, siteInfoFile, climDataFile, targetDataFile, coordFile = None, speciesStrs = ['pine', 'spr', 'bl'], verbose = False):

# This function takes the forest variable data, site info data and climate data 
# files as input and combines these data in one dataframe and saves the data as
# a *.csv file.

	forVarData_df = pd.read_csv(forVarDataFile)
	
	# Read the rest input data:
	siteInfo_df = pd.read_csv(siteInfoFile)
	targetData_df = pd.read_csv(targetDataFile)
	climData_df = pd.read_csv(climDataFile)
	
	# Compute the number of years in the climate data. This is
	# for future possible use, the present default is nrYears = 1:
	climIDs_uniq = pd.unique(climData_df['climID'])
	nrYears = int(climData_df.shape[0]/(climIDs_uniq.shape[0] * 365))
	if verbose:
		print("climData_df.shape = ", climData_df.shape)
		print("nrYears = ", nrYears)
	
	# ===============================================================================
	# 1. Generate output dataframe and fill it with zeros:
	# -------------------------------------------------------------------------------
	# Extract unique site ID's. NOTE that the Pandas function pd.unique() returns  
	# the unique data set in order of appearance in the original data set:
	siteIDs_uniq = pd.unique(forVarData_df['site'])
	
	# Define siteInfo headers (hard coded). Note that Hc and A not included. Note also
	# that the variables: 'SWinit', 'CWinit', 'SOGinit', 'Sinit', 'nLayers', 'nSpecies', 
	# 'soildepth', 'effective field capacity', and 'permanent wilting point' are missing
	# from the output dataframe - these are constant default values for the moment, and
	# do not affect in generating the AI emulator model. The column headers 'climID' and
	# 'siteType' will be added when merging these data from the siteInfo datafarame to
	# the output dataframe:
	aiDataSetHdrs = ['siteID', 'areaID', 'age_pine', 'H_pine', 'D_pine', 'BA_pine', 'age_spr', 'H_spr', 'D_spr', 'BA_spr', 'age_bl', 'H_bl', 'D_bl', 'BA_bl']

	# add the daily climate data headers:
	days = range(1,nrYears*365+1)
	weatherStrs = ['TAir', 'Precip', 'VPD']
	weatherHdrs = [(i + '_' + str(x)) for i in weatherStrs for x in days]

	# and the monthly PAR data headers:
	months = range(1,nrYears*12+1)
	parHdrs = [('PAR_' + str(x)) for x in months]
	
	# Update the output dataframe headers (commented due to the way that the climate data
	# dataframe is computed in step #4 below):
	#aiDataSetHdrs = aiDataSetHdrs + weatherHdrs + parHdrs + ['CO2']
	if verbose:
		print("aiDataSetHdrs = ", aiDataSetHdrs)
	
	# Locate the forest variable data starting index on the output dataframe:
	fsvStartIdx = aiDataSetHdrs.index('age_pine')
	nrForestVars = 4
	
	# ------------------------------------------------------------------------------------
	# Add climate data headers (PAR_1, PAR_2, ..., PAR_365; TAir_1, TAir_2, ...) later ...
	# ------------------------------------------------------------------------------------

	aiDataSet = pd.DataFrame(np.zeros((siteIDs_uniq.shape[0], len(aiDataSetHdrs))), columns = aiDataSetHdrs)
	if verbose:
		print("aiDataSet.shape = ", aiDataSet.shape)
	
	# ===============================================================================
	# 2. Organize forest variable data into a dataframe containing one row per site:
	# -------------------------------------------------------------------------------
	
	# Compute some column indices beforehand (in forVarData_df):
	areaIDidx_fsv = forVarData_df.columns.get_loc('areaID')
	speciesIDidx_fsv = forVarData_df.columns.get_loc('speciesID')
	ageIdx_fsv = forVarData_df.columns.get_loc('age')
	baIdx_fsv = forVarData_df.columns.get_loc('BA')
	if verbose:
		print("areaIDidx_fsv = ", areaIDidx_fsv)
		print("speciesIDidx_fsv = ", speciesIDidx_fsv)
	
	for ii, thisSiteId in enumerate(siteIDs_uniq):
		aiDataSet.loc[ii, 'siteID'] = thisSiteId
		
		# Select the rows of this site:
		forVarDataThisSite = forVarData_df.query('site == ' + str(thisSiteId))
		# if verbose:
			# print("forVarDataThisSite.shape = ", forVarDataThisSite.shape)
		
		# Take the 'areaID' from the first row of this site's dataframe:
		areaIDidx = ['areaID' in i for i in forVarDataThisSite.columns]
		aiDataSet.iloc[ii, aiDataSetHdrs.index('areaID')] = forVarDataThisSite.iloc[0,areaIDidx_fsv]
		
		# There are 'nrThisSiteSpecies' rows in 'forVarDataThisSite' depending on
		# the number of species on the site (i.e. siteIdx = [1,..., nrThisSiteSpecies]):
		for siteIdx in range(forVarDataThisSite.shape[0]):
				# Use for loop to find all the species data in this site, as the
				# number of species may cahnge from the default (= 3):
				for specIdx, speciesStr in enumerate(speciesStrs):
					# SpeciesID numbering starts from 1:
					speciesID = specIdx + 1
					if forVarDataThisSite.iloc[siteIdx, speciesIDidx_fsv] == speciesID:
						# Add the forest variable data for this species to the 
						# corresponding location in the output dataframe row:
						aiDataSet.iloc[ii, fsvStartIdx + (speciesID-1)*nrForestVars:fsvStartIdx + speciesID*nrForestVars] = forVarDataThisSite.iloc[siteIdx, ageIdx_fsv:baIdx_fsv+1]

		# if ii == 20:
			# break

	# ===============================================================================
	# 3. Join the variables 'climID' and 'siteType' from the siteInfo dataframe: 
	# -------------------------------------------------------------------------------

	aiDataSet = aiDataSet.merge(siteInfo_df[['siteID', 'climID', 'siteType']], how='inner', on='siteID')

	# ===============================================================================
	# 4. Join the climate data variables
	#
	# Produce a new dataframe with the climate data on each row using the 'siteIDs_uniq'
	# for loop as in step #2 above. Then concatenate this dataframe with 'aiDataSet'.
	# This way there's no need to compute any 'aiDataSet' column indices.
	# -------------------------------------------------------------------------------

	climDataSetHdrs = weatherHdrs + parHdrs + ['CO2']

	climDataSet = pd.DataFrame(np.zeros((siteIDs_uniq.shape[0], len(climDataSetHdrs))), columns = climDataSetHdrs)
	
	for ii, thisSiteId in enumerate(siteIDs_uniq):
		thisSiteClimID = aiDataSet.iloc[ii, aiDataSet.columns.get_loc('climID')]
		
		# Select the rows of this climate ID from the input dataframe:
		thisClimIDdata = climData_df.query('climID == ' + str(thisSiteClimID))
		
		# Compute the number of obtained rows - this is a multiple of 365 
		# (default = 365):
		nrThisClimDataRows = thisClimIDdata.shape[0]
		
		# Take only the monthly PAR data (discard duplicates):
		PAR = pd.unique(thisClimIDdata['PAR'])
		PAR = PAR.reshape((1, PAR.shape[0]))
		
		# ... and yearly (yes!) CO2 data:
		CO2 = pd.unique(thisClimIDdata['CO2']).reshape((1,nrYears))
		
		# Extract the 
		TAir = thisClimIDdata['TAir'].values.reshape((1, nrThisClimDataRows))
		Precip = thisClimIDdata['Precip'].values.reshape((1, nrThisClimDataRows))
		VPD = thisClimIDdata['VPD'].values.reshape((1, nrThisClimDataRows))
		
		thisClimIDoutData = np.concatenate((TAir, Precip, VPD, PAR, CO2), axis=1)
		climDataSet.iloc[ii,:] = thisClimIDoutData

		# if verbose:
			# print("TAir.shape = ", TAir.shape)
			# print("PAR.shape = ", PAR.shape)
			
		# if ii == 15:
			# break

	# Concatenate the climate data with the output dataframe:
	aiDataSet = pd.concat([aiDataSet, climDataSet], axis = 1)

	# ===============================================================================
	# 5. Join the target data from 'targetData_df' dataframe:
	#
	# This rquires the re-organizing the input dataframe data from each site into
	# one single row (input data has N rows for each site, N = nbr of species) like
	# in step #2, but is somewhat simpler as the target dataframe has rows foe each
	# species (i.e. N rows / site). The species may not be in sorted order, so the
	# input rows/site must be sorted first according to ascending species ID.
	#
	# Generate the re-organized target dataframe firs and then join it with the 
	# output dataframe using siteID as joining key:
	# -------------------------------------------------------------------------------

	# filter the target data input dataframe to exclude the zero rows (missing tree
	# species in site):
	targetData_df = filterDataFrame(targetData_df, ['H > 0', 'D > 0', 'BA > 0'])

	# Compute some column indices beforehand (in targetData_df):
	speciesIDidx_tgt = targetData_df.columns.get_loc('species')
	
	# Create the column headers for the target data output. Note that this produces
	# the species-wise headers for all variables, of which some are not species-wise
	# wise in nature (e.g. GPP). This can be corrected later by manipulating the output
	# table directly, if desired:
	targetHdrsIn = targetData_df.columns.values.tolist()
	targetHdrs = [(i + '_' + str(x) + '_t') for x in speciesStrs for i in targetHdrsIn]
	
	# Remove the headers with 'siteID' or 'siteTpe' (leave the header 'species'
	# as a dummy variable for monitoring correct operation):
	ind = [idx for idx, s in enumerate(targetHdrs) if 'site' in s]
	for idx in ind[::-1]:
		targetHdrs.remove(targetHdrs[idx])
	
	# The number of output target variables equals the number of output dataframe
	# variables divided with the number of species:
	nrTgtVars = int(len(targetHdrs)/len(speciesStrs))
	if verbose:
		print("nrTgtVars = ", nrTgtVars)
	
	# Generate target dataframe:
	targetDataSet = pd.DataFrame(np.zeros((siteIDs_uniq.shape[0], len(targetHdrs))), columns = targetHdrs)

	for ii, thisSiteId in enumerate(siteIDs_uniq):
		# Select the input target data rows of this site:
		targetDataThisSite = targetData_df.query('siteID == ' + str(thisSiteId))
		
		# There are 'nrThisSiteSpecies' rows in 'targetDataThisSite' depending on
		# the number of species on the site (i.e. siteIdx = [1,..., nrThisSiteSpecies]):
		for siteIdx in range(targetDataThisSite.shape[0]):
				# Use for loop to find all the species data in this site, as the
				# number of species may cahnge from the default (= 3):
				for specIdx, speciesStr in enumerate(speciesStrs):
					# SpeciesID numbering starts from 1:
					speciesID = specIdx + 1
					if targetDataThisSite.iloc[siteIdx, speciesIDidx_tgt] == speciesID:
						# Add the target data for this species to the 
						# corresponding location in the output dataframe row:
						targetDataSet.iloc[ii, (speciesID-1)*nrTgtVars:speciesID*nrTgtVars] = targetDataThisSite.iloc[siteIdx, speciesIDidx_tgt:targetDataThisSite.shape[1]]
		
		if verbose:
			print("targetDataSet.shape = ", targetDataSet.shape)
			print("targetHdrs = ", targetHdrs)
			print("targetDataThisSite.shape = ", targetDataThisSite.shape)
			print("targetDataThisSite = ", targetDataThisSite)
	
		# if ii == 10:
			# break

	# Concatenate the target data with the output dataframe:
	aiDataSet = pd.concat([aiDataSet, targetDataSet], axis = 1)
	
	# Join the sampling coordinates if coordinate file supplied. The coordinate
	# file must contain the columns 'siteID', 'coord_x' & 'coord_y'. The coordinates
	# will be joined to 'aiDataSet' dataframe using 'siteID' as joining key: 
	if coordFile is not None:
		samplingCoord_df = pd.read_csv(coordFile)
		aiDataSet = aiDataSet.merge(samplingCoord_df, how='inner', on='siteID')
	
	#return targetDataSet
	return aiDataSet


def unStackDataFrame(dataframe_in, multiIndLabels, level = -1, replaceHdrs = None, fileOut = None):

	# Define multi-index (two levels) from user defined columns:
	index = pd.MultiIndex.from_frame(dataframe_in[multiIndLabels])
	#index = pd.MultiIndex.from_frame(dataframe_in[[multiIndLabels[0], multiIndLabels[1]]])
	dataframe_in.set_index(index, drop=False, inplace=True)
	
	# hdrs_in = dataframe_in.columns.values.tolist()
	# print("len(hdrs_in) = ", len(hdrs_in))
	# print("hdrs_in = ", hdrs_in)
	
	# Remove the index columns (the flag 'Drop' does not seem to work in previous line;
	# lots of people complaining this in the web also):
	dataframe_in = dataframe_in.drop(labels=multiIndLabels, axis='columns')
	#dataframe_in.drop(labels=[multiIndLabels[0], multiIndLabels[1]], axis='columns', inplace=True)
	
	# hdrs_in = dataframe_in.columns.values.tolist()
	# print("len(hdrs_in) = ", len(hdrs_in))
	# print("hdrs_in = ", hdrs_in)
	
	# Unstack the input dataframe on specified level index:
	dataframe_out = dataframe_in.unstack(level=level)
	
	# df_out_hdrs = dataframe_out.columns.values.tolist()
	# print("len(df_out_hdrs) = ", len(df_out_hdrs))
	# print("df_out_hdrs = ", df_out_hdrs)

	# Rename columns:
	hdrs_level0 = pd.unique(dataframe_out.columns.get_level_values(0).tolist())
	if replaceHdrs == None:
		hdrs_level1 = pd.unique(dataframe_out.columns.get_level_values(1).tolist())
	else:
		hdrs_level1 = replaceHdrs
	
	# Compose new headers:
	hdrs_out = [(str(i) + '_' + str(x)) for i in hdrs_level0 for x in hdrs_level1]
	
	# The level 0 header still remains in the dataframe for further use.
	# Add it here:
	firstColHdrs = []

	for thisHdr in multiIndLabels[0:level]:
		firstColHdrs += [thisHdr]

	hdrs_out = firstColHdrs + hdrs_out
	
	# Remove multi-indices (keep level0 column; drop = False):
	dataframe_out.reset_index(drop=False, inplace=True)
	dataframe_out.columns = hdrs_out
	
	if fileOut is not None:
		dataframe_out.to_csv(fileOut, index=False)
	
	return dataframe_out


def unStackDataFrame_twoLevels(dataframe_in, multiIndLabels, level = -1, fileOut = None):

	# Define multi-index (two levels) from user defined columns:
	index = pd.MultiIndex.from_frame(dataframe_in[[multiIndLabels[0], multiIndLabels[1]]])
	dataframe_in.set_index(index, drop=False, inplace=True)
	
	# hdrs_in = dataframe_in.columns.values.tolist()
	# print("len(hdrs_in) = ", len(hdrs_in))
	# print("hdrs_in = ", hdrs_in)
	
	# Remove the index columns (the flag 'Drop' does not seem to work in previous line;
	# lots of people complaining this in the web also):
	dataframe_in.drop(labels=[multiIndLabels[0], multiIndLabels[1]], axis='columns', inplace=True)
	
	# hdrs_in = dataframe_in.columns.values.tolist()
	# print("len(hdrs_in) = ", len(hdrs_in))
	# print("hdrs_in = ", hdrs_in)
	
	# Unstack the input dataframe on specified level index:
	dataframe_out = dataframe_in.unstack(level=level)
	
	# df_out_hdrs = dataframe_out.columns.values.tolist()
	# print("len(df_out_hdrs) = ", len(df_out_hdrs))
	# print("df_out_hdrs = ", df_out_hdrs)

	# Rename columns:
	hdrs_level0 = pd.unique(dataframe_out.columns.get_level_values(0).tolist())
	hdrs_level1 = pd.unique(dataframe_out.columns.get_level_values(1).tolist())

	# Compose new headers:
	hdrs_out = [(str(i) + '_' + str(x)) for i in hdrs_level0 for x in hdrs_level1]
	
	# The level 0 header still remains in the dataframe for further use.
	# Add it here:
	hdrs_out = [multiIndLabels[0]] + hdrs_out
	
	# Remove multi-indices (keep level0 column; drop = False):
	dataframe_out.reset_index(drop=False, inplace=True)
	dataframe_out.columns = hdrs_out
	
	if fileOut is not None:
		dataframe_out.to_csv(fileOut, index=False)
	
	return dataframe_out


def extractPrebasOutputs(multiPrebasOut, multiOutDefFile = None, multiOutFile=None):

	multiPrebasOut_rec = recurse_r_tree(multiPrebasOut)

	# Extract all the separate outputs of multiPrebas:
	# For details, see 'multiPrebas_help.pdf'
	multiOut = multiPrebasOut_rec['multiOut']
	dailyPreles = multiPrebasOut_rec['dailyPRELES']
	GVout = multiPrebasOut_rec['GVout']
	soilC = multiPrebasOut_rec['soilC']
	
	# -----------------------------------------------------------------------
	# 1. Extract the desired variables from multiOut:
	# -----------------------------------------------------------------------
	
	# Read the desired variables from definition file 'multiOutVariables.csv'.
	# Use default file if not speciefied otherwise:
	if multiOutDefFile == None:
		multiOutDefPath = 'C:\\PROJECTS\\2023_ARTISDIG\\WP4\\AI_EMULATOR\\ART_REPO'
		multiOutDefFile = os.path.join(multiOutDefPath, 'multiOutVariables.csv')

	multiOutVarDef_df = pd.read_csv(multiOutDefFile)
	# Extract multiOut array columns headers and the corresponding column indices:
	multiOutExtHdrs = list(multiOutVarDef_df['variable'].values)
	# Add 'Year' as the first column to output dataframe:
	multiOutExtHdrs = ['Year'] + multiOutExtHdrs
	outputvarCols = multiOutVarDef_df['varIndex'].values
	
	nSites = multiOut.shape[0]
	nYears = multiOut.shape[1]
	nSpecies = multiOut.shape[3]
	speciess = range(nSpecies)

	# Create output dataframe:
	multiOutExt = pd.DataFrame(columns = multiOutExtHdrs)

	# Repeat the default vector 'nSites' times:
	# This is very inefficient! To be re-coded later!
	for thisYear in range(nYears):
		for thisSite in range(nSites):
			for thisSpecies in speciess:
				for ii, outVarCol in enumerate(outputvarCols):
					if ii==0:
						bar = np.array(thisYear+1, ndmin=2)
						foo = np.array(multiOut[thisSite,thisYear,outVarCol,thisSpecies,0], ndmin=2)
						foo = np.concatenate((bar, foo), axis=1)
					else:
						foo = np.concatenate((foo, np.array(multiOut[thisSite,thisYear,outVarCol,thisSpecies,0], ndmin=2)), axis=1)

				foo_df = pd.DataFrame(foo, columns = multiOutExtHdrs)
				multiOutExt = pd.concat([multiOutExt, foo_df], ignore_index=True)

	if multiOutFile is not None:
		multiOutExt.to_csv(multiOutFile, index=False)

	return multiOutExt


def add_YMD_to_climData(climDataFile, outFile = None):

	climData_df = pd.read_csv(climDataFile)

	# convert the date strings (column 'date') into
	climData_YMD = dates_to_YMDs(climData_df['date'])

	# Add the columns YEAR, MONTH, DAY to the input dataframe:
	climData_df['YEAR'] = climData_YMD[:,0]
	climData_df['MONTH'] = climData_YMD[:,1]
	climData_df['DAY'] = climData_YMD[:,2]

	# Add also an integer indicating to which part of a month
	# the date belongs (1: DAY < 16; 2: DAY >=16). This is for
	# later computing bi-monthly aggregates:
	bb = np.zeros((climData_YMD.shape[0],1))
	
	for ii in range(climData_YMD.shape[0]):
		bb[ii] = 1 if climData_YMD[ii,2] < 16 else 2
	
	# Compute a bi-month indicator (unique per year):
	biMonth = 10*climData_YMD[:,1]
	biMonth = np.reshape(biMonth, (climData_YMD.shape[0],1))
	climData_df['BI_MONTH'] = biMonth + bb
	
	if outFile is not None:
		climData_df.to_csv(outFile)
	
	return climData_df


def dates_to_YMDs(dates):
    dateOut = np.zeros((dates.shape[0], 3), dtype=int)
    
    for ii in range(dates.shape[0]):
        # The input string is of format: dd.mm.yyyy:
        dayStr, monthStr, yearStr = str(dates[ii]).split(".")
        # Reverse the order of D,M,Y to the output array -> [Y, M, D]
        # Return as integer table:
        dateOut[ii,0] = int(yearStr)
        dateOut[ii,1] = int(monthStr)
        dateOut[ii,2] = int(dayStr)
        #dateOut[ii,0], dateOut[ii,1], dateOut[ii,2] = str(dates[ii]).split(".")
        
    return dateOut




# computeGroupAggregates()
#
# This function computes group summaries (= aggregates) using the
# Pandas groupBy() function.
#
# Inputs:
#
# input_df		(Pandas dataframe) Dataframe containing the input data 
#				and grouping key columns.
# methods		(list of strings) The methods for computing the aggregates.
#				Available methods are ['sum', 'mean', 'std', 'min', 'max']
# keys			(list of strings) The column names of the grouping keys.
# columnsToKeep	(list of strings) The column names of the columns to include 
#				in the output dataframe.
# outFile		(path) The filename (path + filename) of the output (*.csv) file.
#
# sumPosCol		(list of strings) This is a special case. To compute the sum of 
#				positive values only (e.g. to get sort of a temperature sum), 
#				give a list of column names here (default = None). This parameter
#				affects only the 'sum' method operation. 

def computeGroupAggregates(input_df, methods = ['mean'], keys = None, columnsToKeep = None, outFile = None, sumPosCol = None):

	for ii, thisMethod in enumerate(methods):
		if thisMethod == 'sum':
			if sumPosCol is not None:
				input_df_sum = replaceValuesBelowThreshold(input_df, sumPosCol, [0], [0])
			else:
				input_df_sum = input_df
		
			aggregate_df = input_df_sum.groupby(keys).sum()
			
		if thisMethod == 'mean':
			aggregate_df = input_df.groupby(keys).mean()
		if thisMethod == 'std':
			aggregate_df = input_df.groupby(keys).std()
		if thisMethod == 'min':
			aggregate_df = input_df.groupby(keys).min()
		if thisMethod == 'max':
			aggregate_df = input_df.groupby(keys).max()

		# Use all columns, if not specified otherwise:
		if columnsToKeep == None:
			columnsToKeep = aggregate_df.columns
			
		aggregate_df = aggregate_df[columnsToKeep]
			
		# Use the first aggregate df as the output_df basis:
		if ii == 0:
			output_df = aggregate_df
		else:
			output_df = pd.concat([output_df, aggregate_df], axis=1)
			
		# Add summary method string to output column headers:
		outHdrs = [(i + '_' + thisMethod) for i in columnsToKeep]
		output_df = replace_column_names(output_df, columnsToKeep, outHdrs)

	# Remove multi-indices:
	output_df.reset_index(drop=False, inplace=True)

	if outFile is not None:
		output_df.to_csv(outFile, index=False)

	return output_df


# replaceValuesBelowThreshold()
#
# This function replaces the values of a Pandas data frame column(s) 
# that are less than given threshold(s) with specified  new value(s).
# The thresholds and new values may be give as common to all specified 
# columns (as lists with single element), or as individual to the 
# specified data frame columns, in which case the length of thresholds
# and new_values must agree with the length of the columns' name list 
# (= columns).

def replaceValuesBelowThreshold(df, columns, thresholds, new_values):
    # Copy the original DataFrame to avoid modifying it in place
    modified_df = df.copy()

    for i, thisColumn in enumerate(columns):
        if len(thresholds) > 1:
            threshold = thresholds[i]
        else:
            threshold = thresholds[0]
        if len(new_values) > 1:
            new_value = new_values[i]
        else:
            new_value = new_values[0]
        # Use apply and lambda to replace values in the specified column
        modified_df[thisColumn] = df[thisColumn].apply(lambda x: new_value if x < threshold else x)

    return modified_df




		
# retrieveClimateData()
#
# This function reads climate data reveived from
# the Copernicus Data Store (CDS) for the years 2020 - 2100. The 
# retrieved parameters are PAR, TAir, Precip, VPD, and CO2. 
#
# ------------------------------------------------------------------------------
# Input climate data:
#
# The model and the climate scenarios (experiments) for the data
# collected so far (31.10.2023) are:
#
# model = "hadgem3_gc31_ll"
# experiment = "ssp1_2_6"
# experiment = "ssp2_4_5"
# experiment = "ssp5_8_5"
#
# The data has been stored in 20 years periods into *.csv files
# (two files with one year period) named as:
#
# WEATHER_3_1_hadgem3_gc31_ll_ssp2_4_5_yyyy_yyyy_20231024_YMD.csv
#
# with sspx_y_z = [ssp1_2_6, ssp2_4_5, ssp5_8_5], and
# yyyy_yyyy = [2020_2020, 2020_2040, 2021_2040, 2041_2060, 2061_2080, 2081_2100]
#
# The data has been fetched with the function CDSA_api.py/getClimateData(), and
# the columns YEAR MONTH DAY BI_MONTH has been added to the original data files
# by forestVarData_tools.py/add_YMD_to_climData().
#
# The input data file contains the daily climate/weather data for the yyyy_yyyy
# period, and contains the data columns: 
#
# climID date PAR TAir Precip VPD CO2 YEAR MONTH DAY BI_MONTH
#
# --------------------------------------------------------------------------------
# This function will combine data from C separate climate zones (e.g C = 10) 
# out of the total 130 climate zones for a period of Y years (e.g. Y = 25), by
# reading the data from the described *.csv files.
#
# The following rules will apply in randomly collecting the data:
# 1) The climate C zones will be randomly selected (out of 130).
# 2) The year range Y will be randomly selected from the 2020 - 2100 period, and
# independently for each climate zone.
# 3) The original climate zone ID (climID) will be mapped to new integer ID values
# (1 - C) and the original climID will be saved in variable climIDorig (the new 
#climID's will be assigned as climID for compatibility with prebas SW).
# 4) The year range for each climate zone will be saved also
#
# The climate ID's (the original and the new one) and the year range for each 
# climate zone will be returned in a cross-ref table for subsequent use.
#
# inPath		(path) The path to the climate data folder.
#
# nrClimIDs		(int or list of int) Twofold operation:
#				1) If single (int) number given: The number of (random) climate 
#				data zones (climIDs) to retrieve data from.
#				2) If a list of integers given: The climate data zones to retrieve
#				data from (i.e. to force specific climate data zones).
#
# nrYears		(int or list of int) Twofold operation:
#				1) If single (int) number given: The number of years (range) of data 
#				to retrieve for each climate zone.
#				2) If a list of two integers given: The first list element is taken as
#				the starting year, and the second as the end year of the prediction period.
#
# scenarios		(list of strings) The climate scenarios (aka experiments in CDS) that the 
#				input data includes. The strings are used to locate the respective input
#				data file. To force a certain scenario, reduce the list of scenarios to one.
#
# model			(list of strings) The model(s) used to generate the input data. The 
#				strings used to locate the respective input file. To force a certain 
#				model, reduce the list of models to one.
#
#				NOTE: Presently (1.11.2023) only one model used, and the code does not
#				support searching the files per models! To be coded in more models used!
#
# yAll_start	(int) The start year of the input climate data (default = 2020)
#
# yAll_end		(int) The end year of the input climate data (default = 2100)
#
# nrClimIDs_all	(int) Total number of climate data zones in the input data.
#				This value defines the number of all climate data zones that exist 
#				in the input climate data (obtained from CDS)
#
# outFile		(string) Output filename. If given, the climate data will be
#				written to a *.csv file (default = None)
#
# movingAve		(bool) If true, compute 30 day moving average of RAR, and 365
#				day moving average of CO2 (default = False)

def retrieveClimateData(inPath, nrClimIDs = 10, nrYears = 25, 
		scenarios = ['ssp1_2_6', 'ssp2_4_5', 'ssp5_8_5'], 
		models = ['hadgem3_gc31_ll'], yAll_start = 2020, yAll_end = 2100, 
		nrClimIDs_all = 130, 
		outFile = None, 
		movingAve = False,
		verbose = False):

	# Init random generator:
	rng = np.random.default_rng()

	# Derive the given climate data zone IDs:
	if isinstance(nrClimIDs, list):
		# If a list is given, interpret the list items as the (original)
		# climate zone IDs:
		climIDs_orig = nrClimIDs
		nrClimIDs = len(nrClimIDs)
	else:
		# Generate the random set of selected climate data zones.
		climIDs_orig = np.arange(nrClimIDs_all)
		rng.shuffle(climIDs_orig)
		# Take a subset of size 'nrClimIDs' of the shuffled climID's:
		climIDs_orig = climIDs_orig[0:nrClimIDs]

	# Select the models for each nrClimIDs randomly:
	# Note that on 1.11.2023 only one model was available:
	model_idx = np.arange(nrClimIDs)
	rng.shuffle(model_idx)
	model_idx = np.mod(model_idx, len(models), dtype = np.int16)
	
	# Build a pandas dataframe with the produced random models (strings):
	model_df = pd.DataFrame(columns=['model'], index=range(nrClimIDs))
	for ii, thisModelIdx in enumerate(model_idx):
		model_df['model'][ii] = models[thisModelIdx]

	# Select the scenarios for each nrClimIDs randomly:
	scenario_idx = np.arange(nrClimIDs)
	rng.shuffle(scenario_idx)
	scenario_idx = np.mod(scenario_idx, len(scenarios), dtype = np.int16)

	# Build a pandas dataframe with the produced random scenarios (strings):
	scenario_df = pd.DataFrame(columns=['scenario'], index=range(nrClimIDs))
	for ii, thisScenarioIdx in enumerate(scenario_idx):
		scenario_df['scenario'][ii] = scenarios[thisScenarioIdx]

	# Derive the given years:
	if isinstance(nrYears, list):
		# If a list of two items is given, take the first element as the
		# starting year, and the second as the end year of the desired
		# period:
		years_start = np.repeat(nrYears[0], nrClimIDs)
		years_end = np.repeat(nrYears[1], nrClimIDs)
	else:
		# Select the years (nrYears within the range 2020 -2100) randomly:
		years_start = np.arange(yAll_start, yAll_end - nrYears + 1)
		rng.shuffle(years_start)
		# Take a subset of size 'nrClimIDs' of the shuffled starting years:
		years_start = years_start[0:nrClimIDs]
		years_end = years_start + nrYears - 1

	# Save the cross-ref table for [climID climIDs_orig years_start years_end]
	# Note that the variable 'climID' is now the (new) dummy climID to be
	# returned with the retrieved data (Note: climID = 1 - nrClimIDs):
	crossRef_climID = np.array([np.arange(nrClimIDs)+1, climIDs_orig, years_start, years_end]).T
	crossRef_columns = ['climID', 'climID_orig', 'year_start', 'year_end']
	crossRef_climID_df = pd.DataFrame(crossRef_climID, columns = crossRef_columns)

	# Add the models and scenarios as the two last columns:
	crossRef_climID_df['model'] = model_df['model']
	crossRef_climID_df['scenario'] = scenario_df['scenario']

	if verbose:
		print(climIDs_orig)
		print(scenario_idx)
		print(years_start)
		print(years_end)
		print(crossRef_climID_df.shape[0])
		#print(crossRef_climID_df)

	# Get the data for the obtained random climate zones and the random years:
	climateData_df = pd.DataFrame()

	for ii in range(crossRef_climID_df.shape[0]):
		climID_orig = crossRef_climID_df.iloc[ii,crossRef_climID_df.columns.to_list().index('climID_orig')]
		year_start = crossRef_climID_df.iloc[ii,crossRef_climID_df.columns.to_list().index('year_start')]
		year_end = crossRef_climID_df.iloc[ii,crossRef_climID_df.columns.to_list().index('year_end')]
		model = crossRef_climID_df.iloc[ii,crossRef_climID_df.columns.to_list().index('model')]
		scenario = crossRef_climID_df.iloc[ii,crossRef_climID_df.columns.to_list().index('scenario')]
		
		# Compose the year range strings to look for:
		yRangeSearchStrs = composeYrangeSearchStrings(year_start, year_end, verbose = False)
		
		# Find the files associated with the 'model' and 'scenario', and are of the desired processing level ('_YMD.csv)):
		for thisYrangeSearchStr in yRangeSearchStrs:
			thisFile = searchFilesByStrings(inPath, [model, scenario, '_YMD.csv', thisYrangeSearchStr], verbose = False)
			
			# Read the climate data of this 'model' and 'scenario' including 
			# (at least some of) the years of interest:
			data_thisClimID = pd.read_csv(os.path.join(inPath, thisFile[0]))
			
			# Replace the column name 'climID' to 'climID_orig' here:
			data_thisClimID = replace_column_names(data_thisClimID, ['climID'], ['climID_orig'])
			
			# Filter the climate data to include only current climate zone data 'climID_orig'
			# and the years of interest 'year_start' - 'year_end':
			filterStrs = ['climID_orig == ' + str(climID_orig), 'YEAR >= ' + str(year_start), 'YEAR <= ' + str(year_end)]
			data_thisClimID = filterDataFrame(data_thisClimID, filterStrs)
			
			# Compute moving averages for PAR & CO2 as per climID, if desired:
			if movingAve:
				# Compute 30 day moving average of PAR, and 365 day moving average of CO2:
				ma_PAR = movingAvg(data_thisClimID, column = 'PAR', window = 30)
				ma_CO2 = movingAvg(data_thisClimID, column = 'CO2', window = 365)
				
				# Replace the monthly PAR and yearly CO2:
				data_thisClimID['PAR'] = ma_PAR
				data_thisClimID['CO2'] = ma_CO2
		
			# Append the data to output climate dataframe:
			climateData_df = pd.concat([climateData_df, data_thisClimID], ignore_index = True)
			
		if verbose:
			print(yRangeSearchStrs)
			
	# Add the dummy climID's to the output dataframe (keep originals as well for monitoring). 
	# The table 'crossRef_climID_df' will contain the information of the original climID:
	climateData_df = climateData_df.merge(crossRef_climID_df[['climID', 'climID_orig']], how = 'outer', on = 'climID_orig')

	# Reset index
	#climateData_df.reset_index(drop=False, inplace=True)        
	if outFile is not None:
		climateData_df.to_csv(outFile, index=False)
		
	if verbose:
		print(climateData_df.shape[0])
		
	# Compute the transposed (multiSitePrebas requires this) climate
	# variables, and return them as dict:
	climateDataDict = txData4prebas(climateData_df)

	return climateData_df, climateDataDict, crossRef_climID_df


# txData4prebas(climData)
#
# This function takes the climate data frame as input, or 
# optionally reads the *.csv file produced with earlier steps
# of getClimateData() and splits & re-organizes the columns from
# the input data to Pandas dataFrames ready to be input into 
# prebasMultiSiteWrapper.r

def txData4prebas(climData, filterStr = None, verbose = False):

	# Optionally read the pre-processed climate data from file:
	if isinstance(climData, str):
		climData_df = pd.read_csv(climData)
	else:
		climData_df = climData
			
	dataHdrs = climData_df.columns

	# Filter the input geoDataFrame according to 'fiterStr':
	if filterStr is not None:
		climData_df = filterDataFrame(climData_df, filterStr)

	# Use the data naming from prebas software (for transposed data):
	dataNames = ['PAR', 'TAir', 'Precip', 'VPD', 'CO2']

	# Extract unique climID's (first column of input data table):
	climIDs_uniq = climData_df['climID'].unique()
	nrDates = int(climData_df.shape[0]/len(climIDs_uniq))

	if verbose:
		print("climIDs_uniq = ", climIDs_uniq)
		print("nrDates = ", nrDates)

	climateDataOut = OrderedDict()

	# Extract each data type one by one with the 'climID' column: 
	for i, thisHdr in enumerate(dataNames):
		thisdata = climData_df[['climID', thisHdr]][:]
		thisDataOut = np.empty((nrDates,0))
		
		# Filter the data for this climID and
		for thisClimID in climIDs_uniq:
			#print("thisClimID = ", thisClimID)
			thisChunk_df = filterDataFrame(thisdata, 'climID == ' + str(thisClimID))
			thisChunk = thisChunk_df.loc[:,thisHdr].to_numpy()
			
			# Force the the obtained Numpy matrix to have two dimensions
			# (to np.concatenate to succeed):
			thisChunk = np.reshape(thisChunk, (nrDates,1))
			
			# Concatenate the transpose of this data chunk to 'thisDataOut':
			thisDataOut = np.concatenate((thisDataOut, thisChunk), axis=1)
		
		# transpose 'thisDataOut' when all data from all climIDs have been added:
		thisDataOut = np.transpose(thisDataOut)
		#print("thisDataOut.shape = ", thisDataOut.shape)
		
		# Save the output DataFrames into an ordered dict:
		climateDataOut[thisHdr] = pd.DataFrame(thisDataOut)

	# Return the ordered dict with the variables
	# 'PAR', 'TAir', 'Precip', 'VPD', 'CO2'
	return climateDataOut

	

def searchFilesByStrings(inputFolder, searchStrs, verbose = False):

	# List to store selected file names:
	selected_files = []

	# List all files in the target directory:
	for root, dirs, files in os.walk(inputFolder):
		for filename in files:
			# Check if all strings in 'searchStrs' are in the file name:
			if all(s in filename for s in searchStrs):
				selected_files.append(os.path.join(root, filename))

	# Print the list of selected files:
	if verbose:
		for file_path in selected_files:
			print(file_path)
		
	return selected_files


def composeYrangeSearchStrings(year_start, year_end, verbose = False):
    
    # ------------------------------------------------------------
    # NOTE: HARDCODED, HARDCODED, HARDCODED, HARDCODED, HARDCODED!
    # Define here the start and end years of the climate data files.
    # the range has been indicated in the filename by the string 'yStart_yEnd',
    # where the yStart ans yEnd are strings corresponding to the arrays below.
    # For the present climate data  files the set of year range strings are: 
    # '2020_2040', '2041_2060', '2061_2080', 2081_2100' corresponding to the
    # arrays below:
    startYears = np.array([2020, 2041, 2061, 2081], ndmin = 2).T
    endYears = np.array([2040, 2060, 2080, 2100], ndmin = 2).T
    # ------------------------------------------------------------
    
    # Define a Pandas dataframe table for starting and end years of the climate data files:  
    yRanges_df = pd.DataFrame(startYears, columns = ['startYear'])
    yRanges_df['endYear'] = endYears
    
    if verbose:
        print(yRanges_df)
        print("year_start = ", year_start)
        print("year_end = ", year_end)
    
    for i in reversed(range(startYears.shape[0])):
        if startYears[i] <= year_start:
            break
            
    for j in range(endYears.shape[0]):
        if endYears[j] > year_end:
            break
    
    yRanges_search_df = yRanges_df.iloc[i:j+1,:]
    if verbose:
        print(yRanges_search_df.columns.to_list().index('endYear'))
        print(yRanges_search_df)
    
    # Compose search strings (year ranges):
    yRangeSearchStrs = []
    for ii in range(yRanges_search_df.shape[0]):
        yStartStr = str(yRanges_search_df.iloc[ii, yRanges_search_df.columns.to_list().index('startYear')])
        yEndStr = str(yRanges_search_df.iloc[ii, yRanges_search_df.columns.to_list().index('endYear')])
        yRangeSearchStrs.append(yStartStr + '_' + yEndStr)
    
    if verbose:
        print(yRangeSearchStrs)
        
    return yRangeSearchStrs


# concatData_csv2csv()
#
# This function reads data from two *.csv files into Pandas dataframes 
# and concatenates the dataframes by axis=0. The concatenated dataframe 
# will be written to 'outFile' (CSV format).

def concatData_csv2csv(file1, file2, outFile = None):

	# Read data from the CSV files into DataFrames
	df1 = pd.read_csv(file1)
	df2 = pd.read_csv(file2)

	# Concatenate the DataFrames along axis=0 (stacking rows):
	concatenated_df = pd.concat([df1, df2], axis=0)

	# Reset the index of the concatenated DataFrame:
	concatenated_df.reset_index(drop=True, inplace=True)

	# Write the concatenated data to outFile:
	if outFile is not None:
		concatenated_df.to_csv(outFile, index=False)
	
	return concatenated_df

# saveMultiPrebasInputs()
#
# This function saves the input data used for running 'multiPrebas.r'.
# The input dataframes are saved in *.csv format into the output
# folder defined by 'outputFolder'. A unique ID number given with
# 'runID' will be added to the output filenames:

def saveMultiPrebasInputs(runID, outputFolder, forVarData, siteInfo, crossRef_climID):

	forVardataFile = os.path.join(outputFolder,'forVarData_id_' + str(runID) + '.csv')
	forVarData.to_csv(forVardataFile, index=False)
	
	siteInfoFile = os.path.join(outputFolder,'siteInfo_id_' + str(runID) + '.csv')
	siteInfo.to_csv(siteInfoFile, index=False)
	
	crossRefTblFile = os.path.join(outputFolder,'crossRef_climID_id_' + str(runID) + '.csv')
	crossRef_climID.to_csv(crossRefTblFile, index=False)
	

# split_dataframe()
#
# Function to split a DataFrame into chunks
#
# Note that this function returns a list of numpy sub-arrays of size 'chunk_size'.
# The conversion back to Pandas DataFrames has to be done in the calling program.

def split_dataframe(df, chunk_size):
    num_chunks = len(df) // chunk_size
    remainder = len(df) % chunk_size

    # If there's a remainder, add one more chunk
    if remainder > 0:
        num_chunks += 1

    # Use numpy array_split to split the DataFrame
    chunks = np.array_split(df, num_chunks)

    return chunks


# savePrebasClimateInputs()
#
# This function saves the climate data inputs for multiSitePrebasWrapper.r (long 
# time series data (e.g. 25 years) causes the R program to die for some reason. 
# The problem seems to be in the Python / R interface (rpy2 issue?):

def savePrebasClimateInputs(climateDataDict, outPath, runIDstr, verbose = False):

	# Extract the separate climate data variables, and save them in
	# separate *.csv files for prebasMultiSiteWrapper.r to read:
	PAR = climateDataDict['PAR']
	TAir = climateDataDict['TAir']
	Precip = climateDataDict['Precip']
	VPD = climateDataDict['VPD']
	CO2 = climateDataDict['CO2']
	
	# Compose filenames and save the data:
	ff = os.path.join(outPath, 'PAR_' + runIDstr + '.csv')
	PAR.to_csv(ff, index=False, header=False)
	ff = os.path.join(outPath, 'TAir_' + runIDstr + '.csv')
	TAir.to_csv(ff, index=False, header=False)
	ff = os.path.join(outPath, 'Precip_' + runIDstr + '.csv')
	Precip.to_csv(ff, index=False, header=False)
	ff = os.path.join(outPath, 'VPD_' + runIDstr + '.csv')
	VPD.to_csv(ff, index=False, header=False)
	ff = os.path.join(outPath, 'CO2_' + runIDstr + '.csv')
	CO2.to_csv(ff, index=False, header=False)

	if verbose:
		print("PAR.shape = ", PAR.shape)
		print("TAir.shape = ", TAir.shape)
		print("Precip.shape = ", Precip.shape)
		print("VPD.shape = ", VPD.shape)
		print("CO2.shape = ", CO2.shape)


# movingAve()
#
# This function computes a moving average of a dataframe column specified by the user.
#
# df			(Pandas DataFrame) The input DataFrame
# column		(string) The name of the column to compute the moving average to.
# window		(window) The size of the moving average window in rows.
# min_periods	(int) Minimum number of observations in window required to 
#				pandas.DataFrame.rolling have a value; otherwise, result is np.nan.
#				default = 1 (in contrast to pandas.DataFrame.rolling).
#				(see pandas.DataFrame.rolling for details)
# center		(bool) If False, set the window labels as the right edge of the window index.
#				If True, set the window labels as the center of the window index.
#				Default = True (in contrast to pandas.DataFrame.rolling).
#				(see pandas.DataFrame.rolling for details)

def movingAvg(df, column = None, window = None, min_periods=1, center=True):
    
	if column == None or window == None:
		print("movingAve: column or window not specified!")
		return None

	# Calculate the moving average using a loop for each row:
	moving_averages = []
	for i in range(len(df)):
		target_row = i
		moving_avg = df[column].rolling(window, min_periods=1, center=True).mean().iloc[target_row]
		moving_averages.append(moving_avg)
		
	return moving_averages


# combineClimateDataFiles()
#
# A function to combine climate data files / scenario into a sinle *.csv file.
#
# A 'one-time-only' function - saved here for optional later use.

def combineClimateDataFiles(scenarioStr, dateStr, outPath):

	#scenarioStr = 'ssp5_8_5'
	#dateStr = '20231024'

	inPath = 'C:\\PROJECTS\\2023_ARTISDIG\\WP4\\DATA\\WEATHER_DATA_3_1'
	climDataFile_2020_2040 = os.path.join(inPath, 'WEATHER_3_1_hadgem3_gc31_ll_' + scenarioStr + '_2020_2040_' + dateStr + '_YMD.csv')
	climDataFile_2041_2060 = os.path.join(inPath, 'WEATHER_3_1_hadgem3_gc31_ll_' + scenarioStr + '_2041_2060_' + dateStr + '_YMD.csv')
	climDataFile_2061_2080 = os.path.join(inPath, 'WEATHER_3_1_hadgem3_gc31_ll_' + scenarioStr + '_2061_2080_' + dateStr + '_YMD.csv')
	climDataFile_2081_2100 = os.path.join(inPath, 'WEATHER_3_1_hadgem3_gc31_ll_' + scenarioStr + '_2081_2100_' + dateStr + '_YMD.csv')

	print("Reading *.csv files ...")
	climData_2020_2040 = pd.read_csv(climDataFile_2020_2040)
	climData_2041_2060 = pd.read_csv(climDataFile_2041_2060)
	climData_2061_2080 = pd.read_csv(climDataFile_2061_2080)
	climData_2081_2100 = pd.read_csv(climDataFile_2081_2100)

	print("Combining 2020 - 2060 ...")
	# Concatenate the DataFrames along axis=0 (stacking rows):
	climData_2020_2100 = pd.concat([climData_2020_2040, climData_2041_2060], axis=0)
	# Reset the index of the concatenated DataFrame:
	climData_2020_2100.reset_index(drop=True, inplace=True)

	print("Combining 2061 - 2080 ...")
	# Concatenate the DataFrames along axis=0 (stacking rows):
	climData_2020_2100 = pd.concat([climData_2020_2100, climData_2061_2080], axis=0)
	# Reset the index of the concatenated DataFrame:
	climData_2020_2100.reset_index(drop=True, inplace=True)

	print("Combining 2081 - 2100 ...")
	# Concatenate the DataFrames along axis=0 (stacking rows):
	climData_2020_2100 = pd.concat([climData_2020_2100, climData_2081_2100], axis=0)
	# Reset the index of the concatenated DataFrame:
	climData_2020_2100.reset_index(drop=True, inplace=True)

	# save the combined climate data into *.csv file:
	
	#outPath = 'C:\\PROJECTS\\2023_ARTISDIG\\WP4\\DATA\\WD_3_1_COMBINED'
	
	outFile = os.path.join(outPath, 'WEATHER_3_1_hadgem3_gc31_ll_' + scenarioStr + '_2020_2100_' + dateStr + '_YMD.csv')
	print("Saving into: ", outFile)
	climData_2020_2100.to_csv(outFile, index=False)

	print("Done!")
	
	
	
# combinePrebasData()
#
# This function takes the forest variable data, site info data and target data 
# as input and combines these data in one dataframe and saves the data as
# a *.csv file.	
	
def combinePrebasData(forVarData, siteInfoData, climCrossRefData, targetData = None, 
			speciesStrs = ['pine', 'spr', 'bl'], verbose = False, outFile = None):

	# Optionally read the input data from file:
	if isinstance(forVarData, str):
		forVarData_df = pd.read_csv(forVarData)
	else:
		forVarData_df = forVarData
		
	if isinstance(siteInfoData, str):
		siteInfo_df = pd.read_csv(siteInfoData)
	else:
		siteInfo_df = siteInfoData
		
	if isinstance(targetData, str):
		targetData_df = pd.read_csv(targetData)
	else:
		targetData_df = targetData
	
	if isinstance(climCrossRefData, str):
		climCrossRef_df = pd.read_csv(climCrossRefData)
	else:
		climCrossRef_df = climCrossRefData
	
	# ===============================================================================
	# 1. Unstack forest variable data:
	# -------------------------------------------------------------------------------
	
	# Select only the columns to unstack:
	forVarData_dim = forVarData[['site', 'speciesID', 'age', 'H', 'D', 'BA']]
	aiDataSet = unStackDataFrame(forVarData_dim, ['site', 'speciesID'], replaceHdrs = speciesStrs)
	
	# Replace the header 'site' with 'siteID':
	aiDataSet = replace_column_names(aiDataSet, ['site'], ['siteID'])
	
	# ===============================================================================
	# 2. Join the  siteInfo variables: 
	# -------------------------------------------------------------------------------

	aiDataSet = aiDataSet.merge(siteInfo_df, how='inner', on='siteID')

	# ===============================================================================
	# 3. Join the  climCrossRef_df: 
	# -------------------------------------------------------------------------------

	aiDataSet = aiDataSet.merge(climCrossRef_df, how='inner', on='climID')

	# ===============================================================================
	# 4. Join the target data from 'targetData_df' dataframe:
	#
	# This rquires the re-organizing the target dataframe data from each site into
	# one single row. This is done by calling unStackDataFrame() twice: first to
	# arrange species-wise data into single row/site/year, and then the predictions 
	# from years 1 - N into one row/site.
	#
	# Generate the re-organized target dataframe first and then join it with the 
	# output dataframe using siteID as joining key:
	# -------------------------------------------------------------------------------

	if targetData_df is not None:
		# filter the target dataframe to exclude the zero rows (missing tree
		# species in site):
		targetData_df = filterDataFrame(targetData_df, ['H > 0', 'D > 0', 'BA > 0'])

		# Unstack the target data to get data from separate species on the same row.
		# After this unstack there are still several rows for the same site (for
		# several years):
		targets_unStack1 = unStackDataFrame(targetData_df, ['Year', 'siteID', 'species'], replaceHdrs = speciesStrs)

		# Further unstack data to get data from all years on the same row. After this 
		# there is one row/site:
		targets_unstack2 = unStackDataFrame(targets_unStack1, ['siteID', 'Year'])

		aiDataSet = aiDataSet.merge(targets_unstack2, how='inner', on='siteID')

	# Write the combined data to outFile:
	if outFile is not None:
		aiDataSet.to_csv(outFile, index=False)
	
	return aiDataSet
	

def combinePrebasData_dev(forVarData, siteInfoData, climCrossRefData, targetData = None, 
			speciesStrs = ['pine', 'spr', 'bl'], verbose = False, outFile = None):

	# This function takes the forest variable data, site info data and target data 
	# as input and combines these data in one dataframe and saves the data as
	# a *.csv file.

	# Optionally read the input data from file:
	if isinstance(forVarData, str):
		forVarData_df = pd.read_csv(forVarData)
	else:
		forVarData_df = forVarData
		
	if isinstance(siteInfoData, str):
		siteInfo_df = pd.read_csv(siteInfoData)
	else:
		siteInfo_df = siteInfoData
		
	if isinstance(targetData, str):
		targetData_df = pd.read_csv(targetData)
	else:
		targetData_df = targetData
	
	if isinstance(climCrossRefData, str):
		climCrossRef_df = pd.read_csv(climCrossRefData)
	else:
		climCrossRef_df = climCrossRefData
	
	# ===============================================================================
	# 1. Generate output dataframe and fill it with zeros:
	# -------------------------------------------------------------------------------
	# Extract unique site ID's. NOTE that the Pandas function pd.unique() returns  
	# the unique data set in order of appearance in the original data set:
	# siteIDs_uniq = pd.unique(forVarData_df['site'])
	
	# # Extract the forest variable headers from forVarData df columns (not including Hc & A):
	# rmHdrs = ['site', 'Layers', 'speciesID', 'Hc', 'A']
	# forVarHdrs = [item for item in forVarData_df.columns if item not in rmHdrs]
	# nrForestVars = len(forVarHdrs)
	
	# # Add species fostfix to the forest variable headers:
	# # The speciesWiseHdrs should benow (i.e. same species grouped into consequential columns):
	# # ['age_pine', 'H_pine', 'D_pine', 'BA_pine', 'age_spr', 'H_spr', 'D_spr', 'BA_spr', 'age_bl', 'H_bl', 'D_bl', 'BA_bl']
	# speciesWiseHdrs = [(i + '_' + x) for x in speciesStrs for i in forVarHdrs ]

	# aiDataSetHdrs = ['siteID'] + speciesWiseHdrs
	# if verbose:
		# print("aiDataSetHdrs = ", aiDataSetHdrs)
	
	# # Locate the forest variable data starting index on the output dataframe:
	# fsvStartIdx = aiDataSetHdrs.index('age_pine')
	
	# Create an empty output data set:
	# aiDataSet = pd.DataFrame(np.zeros((siteIDs_uniq.shape[0], len(aiDataSetHdrs))), columns = aiDataSetHdrs)
	# if verbose:
		# print("aiDataSet.shape = ", aiDataSet.shape)
	
	# ===============================================================================
	# 2. Organize forest variable data into a dataframe containing one row per site:
	# -------------------------------------------------------------------------------
	
	# Select only the columns to unstack:
	forVarData_dim = forVarData[['site', 'speciesID', 'age', 'H', 'D', 'BA']]
	aiDataSet = unStackDataFrame(forVarData_dim, ['site', 'speciesID'], replaceHdrs = speciesStrs)
	
	# Replace the header 'site' with 'siteID':
	aiDataSet = replace_column_names(aiDataSet, ['site'], ['siteID'])
	
	# # Compute some column indices beforehand (in forVarData_df):
	# speciesIDidx_fsv = forVarData_df.columns.get_loc('speciesID')
	# ageIdx_fsv = forVarData_df.columns.get_loc('age')
	# baIdx_fsv = forVarData_df.columns.get_loc('BA')
	# if verbose:
		# print("speciesIDidx_fsv = ", speciesIDidx_fsv)
	
	# for ii, thisSiteId in enumerate(siteIDs_uniq):
		# aiDataSet.loc[ii, 'siteID'] = thisSiteId
		
		# # Select the rows of this site:
		# forVarDataThisSite = forVarData_df.query('site == ' + str(thisSiteId))
		# # if verbose:
			# # print("forVarDataThisSite.shape = ", forVarDataThisSite.shape)
		
		# # (Take the 'areaID' from the first row of this site's dataframe:)
		# #areaIDidx = ['areaID' in i for i in forVarDataThisSite.columns]
		
		# # There are 'nrThisSiteSpecies' rows in 'forVarDataThisSite' depending on
		# # the number of species on the site (i.e. siteIdx = [1,..., nrThisSiteSpecies]):
		# for siteIdx in range(forVarDataThisSite.shape[0]):
				# # Use for loop to find all the species data in this site, as the
				# # number of species may change from the default (= 3):
				# for specIdx, speciesStr in enumerate(speciesStrs):
					# # SpeciesID numbering starts from 1:
					# speciesID = specIdx + 1
					# if forVarDataThisSite.iloc[siteIdx, speciesIDidx_fsv] == speciesID:
						# # Add the forest variable data for this species to the 
						# # corresponding location in the output dataframe row:
						# aiDataSet.iloc[ii, fsvStartIdx + (speciesID-1)*nrForestVars:fsvStartIdx + speciesID*nrForestVars] = forVarDataThisSite.iloc[siteIdx, ageIdx_fsv:baIdx_fsv+1]

		# if ii == 20:
			# break

	# ===============================================================================
	# 3. Join the  siteInfo variables: 
	# -------------------------------------------------------------------------------

	aiDataSet = aiDataSet.merge(siteInfo_df, how='inner', on='siteID')

	# ===============================================================================
	# 4. Join the  climCrossRef_df: 
	# -------------------------------------------------------------------------------

	aiDataSet = aiDataSet.merge(climCrossRef_df, how='inner', on='climID')

	# ===============================================================================
	# 5. Join the target data from 'targetData_df' dataframe:
	#
	# This rquires the re-organizing the input dataframe data from each site into
	# one single row. This is done by calling unStackDataFrame() twice: first to
	# arrange species-wise data into single row/site/year, and then the predictions 
	# from years 1 - N into one row/site.
	#
	# Generate the re-organized target dataframe first and then join it with the 
	# output dataframe using siteID as joining key:
	# -------------------------------------------------------------------------------

	if targetData_df is not None:
		# filter the target data input dataframe to exclude the zero rows (missing tree
		# species in site):
		targetData_df = filterDataFrame(targetData_df, ['H > 0', 'D > 0', 'BA > 0'])

		# Unstack the target data to get data from separate species on the same row.
		# After this unstack there are still several rows for the same site (for
		# several years):
		targets_unStack1 = unStackDataFrame(targetData_df, ['Year', 'siteID', 'species'], replaceHdrs = speciesStrs)

		# Further unstack data to get data from all years on the same row. After this 
		# there is one row/site:
		targets_unstack2 = unStackDataFrame(targets_unStack1, ['siteID', 'Year'])

		aiDataSet = aiDataSet.merge(targets_unstack2, how='inner', on='siteID')


	# # Compute some column indices beforehand (in targetData_df):
	# speciesIDidx_tgt = targetData_df.columns.get_loc('species')
	
	# # Create the column headers for the target data output. Note that this produces
	# # the species-wise headers for all variables, of which some are not species-wise
	# # wise in nature (e.g. GPP). This can be corrected later by manipulating the output
	# # table directly, if desired:
	# targetHdrsIn = targetData_df.columns.values.tolist()
	# targetHdrs = [(i + '_' + str(x) + '_t') for x in speciesStrs for i in targetHdrsIn]
	
	# # Remove the headers with 'siteID' or 'siteTpe' (leave the header 'species'
	# # as a dummy variable for monitoring correct operation):
	# ind = [idx for idx, s in enumerate(targetHdrs) if 'site' in s]
	# for idx in ind[::-1]:
		# targetHdrs.remove(targetHdrs[idx])
	
	# # The number of output target variables equals the number of output dataframe
	# # variables divided with the number of species:
	# nrTgtVars = int(len(targetHdrs)/len(speciesStrs))
	# if verbose:
		# print("nrTgtVars = ", nrTgtVars)
	
	# # Generate target dataframe:
	# targetDataSet = pd.DataFrame(np.zeros((siteIDs_uniq.shape[0], len(targetHdrs))), columns = targetHdrs)

	# for ii, thisSiteId in enumerate(siteIDs_uniq):
		# # Select the input target data rows of this site:
		# targetDataThisSite = targetData_df.query('siteID == ' + str(thisSiteId))
		
		# # There are 'nrThisSiteSpecies' rows in 'targetDataThisSite' depending on
		# # the number of species on the site (i.e. siteIdx = [1,..., nrThisSiteSpecies]):
		# for siteIdx in range(targetDataThisSite.shape[0]):
				# # Use for loop to find all the species data in this site, as the
				# # number of species may cahnge from the default (= 3):
				# for specIdx, speciesStr in enumerate(speciesStrs):
					# # SpeciesID numbering starts from 1:
					# speciesID = specIdx + 1
					# if targetDataThisSite.iloc[siteIdx, speciesIDidx_tgt] == speciesID:
						# # Add the target data for this species to the 
						# # corresponding location in the output dataframe row:
						# targetDataSet.iloc[ii, (speciesID-1)*nrTgtVars:speciesID*nrTgtVars] = targetDataThisSite.iloc[siteIdx, speciesIDidx_tgt:targetDataThisSite.shape[1]]
		
		# if verbose:
			# print("targetDataSet.shape = ", targetDataSet.shape)
			# print("targetHdrs = ", targetHdrs)
			# print("targetDataThisSite.shape = ", targetDataThisSite.shape)
			# print("targetDataThisSite = ", targetDataThisSite)
	
		# # if ii == 10:
			# # break

	# # Concatenate the target data with the output dataframe:
	# aiDataSet = pd.concat([aiDataSet, targetDataSet], axis = 1)
	# Write the concatenated data to outFile:
	
	if outFile is not None:
		aiDataSet.to_csv(outFile, index=False)
	
	return aiDataSet




	
def saveRunInfo(runID, outputFolder, wrapperDict):

	outFile = os.path.join(outputFolder, 'wrapperInfo_' + str(runID) + '.csv')

	wrapperDict_df = pd.DataFrame.from_dict(wrapperDict, orient='index')
	wrapperDict_df
	wrapperDict_df.to_csv(outFile, header=False)
	

# readForestDistrClimIDs()
#
# This funclion reads the climate ID's (climID) of the climate data
# zones associated with a given forestry district (identified by 
# fDistStr). The input folder (inpath) must contain the corresponding
# Excel file with the column 'gridcode_1', that lists the climate
# ID's. These Excel files are the result from ArcGIS 'Spatial Join'
# process, associating the climate data zones within 150 km of the
# forestry district border with the forest district.

def readForestDistrClimIDs(fDistStr = 'PiMa', inPath = None):

	inFile = os.path.join(inPath, 'Border_' + fDistStr + '_SpatialJoin.xlsx')
	input_df = pd.read_excel(inFile)

	climIDs = input_df['gridcode_1'].values.tolist()

	return climIDs




def composePrebasSubSet(inPath = None, subSetHdrFile = None, outFile = None, runIDs = None, subIdx = 10, buildStratVarFile = False, verbose = False):

	#outFile  = os.path.join(inPath, 'prebasdata_subSet_1.csv')

	if inPath is None:
		inPath = 'C:\\PROJECTS\\2023_ARTISDIG\\WP4\\AI_EMULATOR\\D_3_1_25Y\\ALL_AREAS'
	if runIDs is None:
		runIDs = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000]
	if subSetHdrFile is None:
		subSetHdrFile = os.path.join(inPath, 'subSet_1_headers.csv')

	subSetHdrs_df = pd.read_csv(subSetHdrFile)
	subSetHdrs = subSetHdrs_df['Header'].values

	nrVectors = 0
	nrSites = 0

	years = np.empty((0,2))

	for ii, thisRunID in enumerate(runIDs):
		
		for jj in range(subIdx):
			thisRunID_iter = thisRunID + jj
			if verbose:
				print(thisRunID_iter)
			#print(thisRunID_iter)
			thisFDdataSetFile = os.path.join(inPath, 'prebasData_id_' +  str(thisRunID_iter) + '.csv')
			thisFDdataSet = pd.read_csv(thisFDdataSetFile)
			
			
			# Extract the specified subset of columns from the input data set:
			thisFDdataSet = thisFDdataSet[subSetHdrs]
		
			nrVectors += thisFDdataSet.shape[0]
			
			# Add 'runID' and 'runID_i' to output dataframe:
			runID_values = np.repeat(thisRunID, thisFDdataSet.shape[0], axis=0)
			runID_i_values = np.repeat(thisRunID_iter, thisFDdataSet.shape[0], axis=0)

			thisFDdataSet['runID'] = runID_values
			thisFDdataSet['runID_i'] = runID_i_values
		
			# record the years also (for monitoring):
			years = np.concatenate((years, thisFDdataSet[['year_start','year_end']].values))
			
			# Add indicator variables exst_pine, exst_spr and exst_bl indicating if
			# the corresponding species is existing (= 1) or absent (= 0) at the stand:
			# Copy the age_xxx variables as the new data columns:
			thisFDdataSet['exst_pine'] = thisFDdataSet['age_pine'].values
			thisFDdataSet['exst_spr'] = thisFDdataSet['age_spr'].values
			thisFDdataSet['exst_bl'] = thisFDdataSet['age_bl'].values

			# Fill the NaN's of the new columns with zeros:
			values = {"exst_pine": 0, "exst_spr": 0, "exst_bl": 0}
			thisFDdataSet = thisFDdataSet.fillna(value = values)

			# Replace values in the new columns that are greater than 0 with 1:
			thisFDdataSet.loc[thisFDdataSet['exst_pine'] > 0, 'exst_pine'] = 1
			thisFDdataSet.loc[thisFDdataSet['exst_spr'] > 0, 'exst_spr'] = 1
			thisFDdataSet.loc[thisFDdataSet['exst_bl'] > 0, 'exst_bl'] = 1
			
			if ii + jj == 0:
				allDataSet = thisFDdataSet
			else:
				allDataSet = pd.concat([allDataSet, thisFDdataSet], axis = 0, ignore_index=True)
			
		# Record the number of unique sites from the last file of this forestry district:
		nrSites += thisFDdataSet.shape[0]

	if verbose:
		print("allDataSet.shape = ", allDataSet.shape)
		print("nrVectors = ", nrVectors)
		print("nrSites = ", nrSites)
		print("years.shape = ", years.shape)

	if outFile is not None:
		if verbose:
			print("Saving ", outFile)
		allDataSet.to_csv(outFile, index=False)
		
	if buildStratVarFile:
		if verbose:
			print("Producing stratification variable file ... ", outFile)
		# Produce data set (pd.DataFrame + *.csv file) with stratification variable.
		# The split will be made by site basis, i.e. the duplicates of same site 
		# removed first:
		stratVarDataSet = allDataSet[['siteID', 'age_pine', 'age_spr', 'age_bl']].drop_duplicates(subset=['siteID'])
		
		# Replace NaN's with zeros in stratVarDataSet:
		stratVarDataSet.fillna(value=0, inplace=True)
		
		# Compute the stratification variable as: stratVar = age_bl + 150 * age_spr + 150*160*age_pine:
		stratVarDataSet['stratVar'] = stratVarDataSet['age_bl'].values + 150*stratVarDataSet['age_spr'].values + 150*160*stratVarDataSet['age_pine'].values
		
		stratvarFile = os.path.join(inPath, 'stratVar.csv')
		if verbose:
			print("Saving ", stratvarFile)
		stratVarDataSet.to_csv(stratvarFile, index=False)

	if verbose:
		print("composePrebasSubSet() done! ", stratvarFile)
		
	return allDataSet


# replaceMissingData()
#
# This routine replaces the missing data cells (missing species data) with
# dummy data.
# 
# The missing species NULL data will be replaced with actual data sampled 
# from the whole data set randomly, so that the distribution of the whole 
# data set does not change significantly. This way the data  used for 
# model training does not induce bias to the model due to distorted input
# data distributions.
#
# Input parameters:

def replaceMissingData(inFile, inputVars = None, targetVars = None, speciess = None, nYears = 25, outFile = None, verbose = False):

	#inputVars = ['age_pine', 'age_spr', 'age_bl', 'H_pine', 'H_spr', 'H_bl', 'D_pine', 'D_spr', 'D_bl', 'BA_pine', 'BA_spr', 'BA_bl']
	if inputVars is None:
		inputVars = ['age', 'H', 'D', 'BA']

	if targetVars is None:
		targetVars= ['H', 'D', 'BA', 'npp', 'ET', 'V', 'GGrowth', 'GPPtrees', 'NEP']

	if speciess is None:
		speciess = ['pine', 'spr', 'bl']

	# Produce target variable names (add the year string to given variable name(s)):
	targetVarCols = [(i + '_' + str(x+1) + '.0') for i in targetVars for x in range(nYears)]

	prebasData = pd.read_csv(inFile)

	# Extract the species data presence/absence indicator variables:
	indTbl = prebasData[['exst_pine', 'exst_spr', 'exst_bl']]

	# Init random generator:
	rng = np.random.default_rng()

	# Extract non-NULL data to sample the replacement data from:
	for thisSpecies in speciess:
		inputVars_thisSpec = [(i + '_' + thisSpecies) for i in inputVars]
		inputVarColIdx = [idx for idx, hdr in enumerate(prebasData.columns) if hdr in inputVars_thisSpec]
		
		tgtVarCols_thisSpec = [(i + '_' + thisSpecies + '_' + str(x+1) + '.0') for i in targetVars for x in range(nYears)]
		tgtVarColIdx = [idx for idx, hdr in enumerate(prebasData.columns) if hdr in tgtVarCols_thisSpec]
		
		replaceData_thisSpec = prebasData[inputVars_thisSpec+tgtVarCols_thisSpec]

		# remove NaN's from this species replacement data:
		replaceData_thisSpec = replaceData_thisSpec.dropna(axis=0)
		if verbose:
			print("prebasData.shape = ", prebasData.shape)
			print("replaceData_thisSpec.shape = ", replaceData_thisSpec.shape)
			#print(replaceData_thisSpec.head())

		# Re-organize the replacement data into random order:
		repl_idx = np.arange(replaceData_thisSpec.shape[0])
		rng.shuffle(repl_idx)
		replaceData_thisSpec = replaceData_thisSpec.iloc[repl_idx, :]
		
		# Sort the prebasData according to this species presence/absence column:
		precAbscol = 'exst_' + thisSpecies
		prebasData.sort_values(by=[precAbscol], inplace=True, ignore_index=True)
		
		# Get the indices of the NULL value rows (for this species data)
		# These are the first len(nullIdx) rows in the sorted dataframe:
		nullIdx = prebasData.index[prebasData[precAbscol]==0]
		#nullIdx = prebasData.index[prebasData[precAbscol]==0].tolist()
		
		if verbose:
			print("len(nullIdx) = ", len(nullIdx))

		# Select the first len(nullIdx) rows of replaceData_thisSpec
		replaceData_thisSpec = replaceData_thisSpec.iloc[nullIdx,:]
		
		prebasData.iloc[nullIdx, inputVarColIdx] = replaceData_thisSpec[inputVars_thisSpec]
		prebasData.iloc[nullIdx, tgtVarColIdx] = replaceData_thisSpec[tgtVarCols_thisSpec]
		
	if outFile is not None:
		prebasData.to_csv(outFile, sep = ',', index = False)

	return prebasData







# buildDataSet4AI()
#
# This function reads the combined Prebas data set (i.e. from
# combinePrebasData() or composePrebasSubSet()) into a yet more
# combined data set including the climate data corresponding to
# each data row in the Prebas data table.

def buildDataSet4AI(prebasData, climData, verbose = False):

	# Optionally read the input data from file:
	if isinstance(prebasData, str):
		prebasData_df = pd.read_csv(prebasData)
	else:
		prebasData_df = prebasData
		
	if isinstance(climData, str):
		climData_df = pd.read_csv(climData)
	else:
		climData_df = climData

	# STAND BY, FOR NOW ... ??/ttehas 19.12.2023
	# INSTEAD INCLUDE THE COMBINATION OF THE CLIMATE DATA & PREBAS
	# DATA INTO THE __getitem__ SECTION OF THE PyTorch data set
	# definition. 




# xlsx2csv(xlsxFile, csvFile)
#
# Read data from Excel file, and save it in *.csv format.

def xlsx2csv(xlsxFile, csvFile):

	input_df = pd.read_excel(xlsxFile)
	input_df.to_csv(csvFile, index=False)


# plotPredictionData()
#
# This function plots the Prebas predictions as solid line graphs.
# The input file fomat is as in:
#
# C:\PROJECTS\2023_ARTISDIG\WP4\AI_EMULATOR\D_3_1_25Y\ALL_AREAS
# prebasData_id_1000.xlsx
#
# Inputs:
# 
# inFile		(string) The input *.csv file (path + name)
# siteIDs		(list of int) List of the sites (field plots) to plot
#				the data for. Each site will be plotted on its own graph.
# variables		(list of strings) A list of substrings to specify the 
#				variables to show. E.g. if substring 'npp' is given,
#				then the npp values for all defined species will be
#				plotted on the same graph.
# species		(list of strings) The species selected to plot.

def plotPredictionData(inFile, siteIDs, variables, species):
    
	prebasData = pd.read_csv(inFile)

	# Filter input data to include the sites in siteIDs:
	prebasData_f = filterDataFrame(prebasData, ['siteID in @subSet'], subSet = siteIDs)

	legendList = []

	mon_vars = ['age_pine', 'age_spr', 'age_bl', 'H_pine', 'H_spr', 'H_bl', 'BA_pine', 'BA_spr', 'BA_bl']
	#monVarStr = "[Age, H, BA]: "

	ccolor = []
	#for ind, thisSpecies in enumerate(species):
	#	if thisSpecies == 'pine':
	#		ccolor += ['blue']
	#	if thisSpecies == 'spr':
	#		ccolor += ['green']
	#	if thisSpecies == 'bl':
	#		ccolor += ['red']
			
	#ccolor = ['blue', 'red', 'green']

	nrSites = prebasData_f.shape[0]
	for jj, thisVariable in enumerate(variables):
		# Locate columns containing the specified substring:
		matching_cols = [col for col in prebasData_f.columns if thisVariable in col]
		
		for ii in range(nrSites):
			# plot separate figure for each site
			fig, ax = plt.subplots()
			
			thisDataRow = prebasData_f.iloc[ii, :]
			thisSiteID = prebasData_f.iloc[ii, 0]
			
			thisScenario = thisDataRow['scenario']
			siteType = thisDataRow['siteType']
			year_start = thisDataRow['year_start']
			year_end = thisDataRow['year_end']
			
			for ss, thisSpecies in enumerate(species):
				if thisSpecies == 'pine':
					ccolor = 'blue'
				if thisSpecies == 'spr':
					ccolor = 'green'
				if thisSpecies == 'bl':
					ccolor = 'red'
					
				# Extract the desired species columns:
				monVarStr = thisVariable + '_' + thisSpecies
			
				matching_cols2 = [col for col in matching_cols if thisSpecies in col]
				thisData = thisDataRow[matching_cols2].values
				thisData = np.array(thisData)
				#print(thisData)
				
				years = np.arange(1, thisData.size + 1)
				years2 = np.arange(year_start, year_start + thisData.size)
				#print(years2)
				ax.plot(years2, thisData)
				
				#if not np.isnan(thisData):
				legendList += [thisSpecies]
				#ax.plot(doy, 20*VPD.loc[climID-1, 0:nYears*365-1])
				#ax.plot(doy, TAir.loc[climID-1, 0:nYears*365-1])
				#ax.plot(doy, SW_div10)
				#ax.legend(['PAR', 'VPDx20', 'TAir'])
				
				# plot the scenario:
				monVar_cols = [col for col in mon_vars if thisSpecies in col]
				#str(thisDataRow[monVar_cols].to_list)
				plt.text(year_end-10, thisData[-1]+0.03*thisData[-1], monVarStr + str(thisDataRow[monVar_cols].values), color = ccolor, fontsize = 'x-small')
				
			
			ax.set_title('siteID: ' + str(thisSiteID) + '  ' + thisScenario + ' SiteType: ' + str(siteType)) 
			ax.legend(legendList)
			ax.set_xlabel('Year', fontsize=10)
			ax.set_ylabel(thisVariable, fontsize=10)



# Some of the multiPrebas predictions seem to be erroneous, so
# that the three height (H_pine, H_spr or H_bl) flattens out after
# ten years or so, or is constant throughout all the 25 year period.
#
# These data rows will be filtered out using an indicator created
# by this routine. The routine assumes that if the tree height at
# 25 years is identical to the mean of the preceding Y years (Y =
# input parameter; default Y = 9), then the data row is erroneous.

def createConstantPredictionIndicator(inFile, outFile = None, constY = 9):

	prebasData = pd.read_csv(inFile)
	prebasDataOut = prebasData

	# Define prediction (target) variable sub-strings (the ending underscore
	# is unique to the target variables):
	treeHeightVars = ['H_pine_', 'H_spr_', 'H_bl_']
	nYears = 25

	for ii, thisVar in enumerate(treeHeightVars):
		indHdr = thisVar + 'err'
		thisSpeciesErrData = pd.DataFrame(columns=[indHdr])
		
		# Select the 'constY' years' data from all this species columns:
		subSetCols = [thisVar+str(ii)+'.0' for ii in range(nYears-constY,nYears+1)]

		# Compute the error indicator column (if = 0, then -> error):
		thisSpeciesErrData[indHdr] = prebasData[subSetCols[-1]].values - prebasData[subSetCols[0:-1]].mean(axis=1).values
		prebasDataOut = pd.concat([prebasDataOut, thisSpeciesErrData], axis=1)

	# Modify filter variables H_xxx_err, xxx = [pine, spr, bl] by replacing
	# NaN's with value 9999:
	df2 = pd.DataFrame(np.ones((prebasDataOut.shape[0], 3))*9999, columns=['H_pine_err', 'H_spr_err', 'H_bl_err'])
	prebasDataOut = prebasDataOut.fillna(df2)

	if outFile is not None:
		prebasDataOut.to_csv(outFile, sep = ',', index = False)

	return prebasDataOut			


# replace_cells()
#
# This function replaces the values of specified Pandas data frame cells.  
# The row indices of the replaced cells are determined from another column of 
# the same data frame having a zero value. The column indices of the replaced 
# cells are specified by user specified header strings. The replacement values 
# come from another data frame (replacement_df).
#
# NOTE: The replacement_df must have as many rows as the input_df.
			
def replace_cells(input_df, condition_column, target_columns, replacement_df):
    # Make a copy of the original DataFrame
    modified_df = input_df.copy()
    
    # Filter rows based on the condition column
    mask = df[condition_column] == 0
    
    # Replace values in the target columns with values from the replacement DataFrame
    modified_df.loc[mask, target_columns] = replacement_df[target_columns]
    
    return modified_df



			
			
def rda_to_npy(file_name, vars = None):
    """
    Convert an .rda object from the R programming language
    to a .npy object for use with numpy
    
    The variable(s) saved in the *.rda file must be listed in parameter 'vars'.
    
    The output is an ordered dict with variables 'vars'.
    
    """
       
    robjects.r["load"](file_name)
    
    dataOut = OrderedDict()
    
    for ii, thisVar in enumerate(vars):
        thisMatrix = robjects.r[thisVar]
        thisMatrix_np = np.array(thisMatrix)
        
        # Save the output DataFrames into an ordered dict:
        dataOut[thisVar] = thisMatrix_np
    
    return dataOut
