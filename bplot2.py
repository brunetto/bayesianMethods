#!/usr/bin/env python
# -*- coding: utf8 -*- 

import sys
import numpy as np
import veusz.embed as ve
import matplotlib.pyplot as plt

def sm_hist(data, delta=5, n_bin=None, range_=None):
	dataMin = np.floor(data.min())
	dataMax = np.ceil(data.max())
	n_bin = np.ceil(1.*(dataMax-dataMin) / delta)
	range_ = (dataMin, dataMin + n_bin * delta)
	counts, bin_edges = np.histogram(data, n_bin, range_, density = False)
	counts = np.hstack((np.array([0]), counts, np.array([0])))
	bin_edges = np.hstack((bin_edges[0], bin_edges))
	return counts, bin_edges

def sm_hist2(data, delta=5):
	dataMin = np.floor(data.min())
	dataMax = np.ceil(data.max())
	n_bin = np.ceil(1.*(dataMax-dataMin) / delta) + 1
	idxs = ((data  - dataMin) / delta).astype(int)
	counts = np.zeros(n_bin+1) # n_bin+1: see last line before return
	edges = np.hstack((dataMin, np.arange(dataMin, dataMax+delta, delta)))#changed from dataMax+1
	for idx in idxs: 
		counts[idx+1] += 1# added +1, check
	counts[-1] = 0 # to close the sm_hist line in veusz, but it's already zero
	return counts, edges

def findLevel(hist2D, which):
	"""
	Find the level (number of counts) containing a `which` fraction of the 
	total area. Start counting from the highest counts.
	"""
	if which < 0 or which > 1:
		print "Wrong percentile, exit!"
		sys.exit(1)
	# Find index in the sorted flatted 2d hist corresponding to a fraction `which`
	# of the area
	id_ = np.searchsorted(np.cumsum(np.sort(np.ravel(hist2D))[::-1])/hist2D.sum(), which) 
	# Find value of the counts level containing that area
	level = np.sort(np.ravel(hist2D))[::-1][id_]
	return level


def to_json(o, level=0):
	"""
	Provided [here](http://stackoverflow.com/questions/10097477/python-json-array-newlines) 
	by [Jeff Terrace](http://jeffterrace.com/).
	Brunetto Ziosi added the sorted.
	"""
	INDENT = 4
	SPACE = " "
	NEWLINE = "\n"

	ret = ""
	if isinstance(o, dict):
		ret += "{" + NEWLINE
		comma = ""
		for k,v in iter(sorted(o.iteritems())):
			ret += comma
			comma = ",\n"
			ret += SPACE * INDENT * (level+1)
			ret += '"' + str(k) + '":' + SPACE
			ret += to_json(v, level + 1)

		ret += NEWLINE + SPACE * INDENT * level + "}"
	elif isinstance(o, basestring):
		ret += '"' + o + '"'
	elif isinstance(o, list):
		ret += "[" + ",".join([to_json(e, level+1) for e in o]) + "]"
	elif isinstance(o, bool):
		ret += "true" if o else "false"
	elif isinstance(o, int):
		ret += str(o)
	elif isinstance(o, float):
		ret += '%.7g' % o
	elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
		ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
	elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
		ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
	else:
		raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
	return ret

class dataO(object):
	def __init__(self, name, data=None):
		self.name = name
		self.label = name
		if data is not None:
			self.idxs = data["idx"]
			self.data = data["data"]
	def stats(self):
		self.mean = self.data.mean()
		self.std = self.data.std()
		return self.mean, self.std
	def freqs(self, delta=None):
		if delta == None:
			n_bin = 50
			self.delta = ((self.data.max()-self.data.min()) / (1.*n_bin))
		else:
			self.delta = delta
		if self.delta == 0:
			print "Zero delta, exit"
			sys.exit(1)
		self.counts, self.bins = sm_hist(self.data, delta=self.delta)
		self.counts = self.counts / (1.*self.counts.sum()*self.delta)
	def percentile(self, which):
		return np.percentile(self.data, which)
	def myPercentile(self, which):
		# Wikipedia definition
		idx = int(np.ceil((which / 100.) * self.data.size))
		return np.sort(data["s"].data)[idx]

infile = sys.argv[1]

document = ve.Embedded("doc_1")
histPage = document.Root.Add('page')
contourPage = document.Root.Add('page')

# Importare nell’ambiente grafico files i files di output di JAGS.
idxType = [("var", "|S30"), ("start","<i8"), ("stop", "<i8")]
dataType = [("idx", "<i8"), ("data", "<f8")]

idxs = np.genfromtxt("CODAindex.txt", dtype=idxType)

dataAll = np.genfromtxt(infile, dtype=dataType)

print idxs

data = {}

for var in idxs:
	if data.has_key(var[0]):
		print var[0], " duplicated! Exit!"
		sys.exit(1)
	data[var[0]] = dataO(var[0], dataAll[var["start"]-1:var["stop"]])

histGridRows = len(idxs)
histGridColumns = 2

histGrid = histPage.Add('grid', autoadd=False, rows = histGridRows, columns = histGridColumns,
							bottomMargin='2cm',
							leftMargin='2.5cm'
							)

#contourGridRows = 1
#contourGridColumns = 1

#contourGrid = contourPage.Add('grid', autoadd=False, rows = gridRows, columns = gridColumns,
							#bottomMargin='2cm',
							#leftMargin='2.5cm'
							#)

dataOut = {}

percentile = 95

for key in data.keys():
	# Calcolare la media e deviazione standard di una sequenza di valori (p.e. s del file allegato
	# ha media 28.55 e standard deviation 16.5)
	print key, data[key].stats()
	
	# Calcolare l’intervallo pi ́u corto che racchiude l’x % dei valori (p.e. s del file allegato ha
	# un intervallo al 95 % che va da 0 a 56)
	print "Compute histogram"
	data[key].freqs()
	
	print "Create json data"
	#dataOut[data[key].name + ":" + "label"] = data[key].label
	dataOut[data[key].name + ":" + "idxs"] = data[key].idxs
	dataOut[data[key].name + ":" + "data"] = data[key].data
	dataOut[data[key].name + ":" + "mean"] = [data[key].mean]
	dataOut[data[key].name + ":" + "std"] = [data[key].std]
	dataOut[data[key].name + ":" + "percentile:" + str(percentile)] = [data[key].percentile(percentile)]
	dataOut[data[key].name + ":" + "bins"] = data[key].bins
	dataOut[data[key].name + ":" + "counts"] = data[key].counts
	
	# Trace
	graph = histGrid.Add('graph', autoadd=False)
	xAxis = graph.Add('axis', name='x', label = "")
	yAxis = graph.Add('axis', name='y', label = "")
	xy = graph.Add('xy')
				
	dataNameX = data[key].name + ":" + "idxs"      
	dataNameY = data[key].name + ":" + "data"
	document.SetData(dataNameX, data[key].idxs)
	document.SetData(dataNameY, data[key].data)
	#xy.xData.val = dataNameX
	#xy.yData.val = dataNameY

	# Histogram
	graph = histGrid.Add('graph', autoadd=False)
	xAxis = graph.Add('axis', name='x', label = "")
	yAxis = graph.Add('axis', name='y', label = "")
	xy = graph.Add('xy',
				marker = 'none',
                PlotLine__steps = u'left',
                )
				
	dataNameX = data[key].name + ":" + "bins"      
	dataNameY = data[key].name + ":" + "counts"
	document.SetData(dataNameX, data[key].bins)
	document.SetData(dataNameY, data[key].counts)
	xy.xData.val = dataNameX
	xy.yData.val = dataNameY

### http://python4mpia.github.io/intro/quick-tour.html
##hist2D, xedges, yedges = np.histogram2d(data["s"].data, data["bkg"].data, bins=50)
##hist2D = np.transpose(hist2D)

### Colormap
##graph = contourGrid.grid.Add('graph', autoadd=False)
##xAxis = graph.Add('axis', name='x', label = "")
##yAxis = graph.Add('axis', name='y', label = "")
##xy = graph.Add('xy')
			
##dataNameX = data[key].name + ":" + "idxs"      
##dataNameY = data[key].name + ":" + "data"
##document.SetData(dataNameX, data[key].idxs)
##document.SetData(dataNameY, data[key].data)
##xy.xData.val = dataNameX
##xy.yData.val = dataNameY

##np.savetxt("joint.csv", hist2D, delimiter=",")

###plt.pcolormesh(xedges, yedges, hist2D, cmap=plt.cm.gray)

### Find levels
##for level in [0.68, 0.95]:
	##threshold = findLevel(hist2D, level)
	##tmp = plt.contour(hist2D, extent=[xedges.min(),xedges.max(),yedges.min(),yedges.max()], levels=[threshold]).collections[0].get_paths()
	##dataOut["level_"+str(level)] = [int(threshold)]
	
	##graph = contourGrid.grid.Add('graph', autoadd=False)
	##xAxis = graph.Add('axis', name='x', label = "")
	##yAxis = graph.Add('axis', name='y', label = "")
	
	##for idx in range(len(tmp)):
		##dataOut["level_"+str(level)+"_contour_x_"+str(idx)] = tmp[idx].vertices[:,0]
		##dataOut["level_"+str(level)+"_contour_y_"+str(idx)] = tmp[idx].vertices[:,1]
		##xy = graph.Add('xy')		
		
		##dataNameX = "level_"+str(level)+"_contour_x_"+str(idx)
		##dataNameY = "level_"+str(level)+"_contour_y_"+str(idx)
		##document.SetData(dataNameX, tmp[idx].vertices[:,0])
		##document.SetData(dataNameY, tmp[idx].vertices[:,1])
		##xy.xData.val = dataNameX
		##xy.yData.val = dataNameY
		

###plt.show()

print "store to json"
out_file = open(infile + "bplots.json","w")
out_file.write(to_json(dataOut))
out_file.flush()
out_file.close()

document.Save(infile + "plot.vsz")
document.Export(infile + "plot.pdf")
