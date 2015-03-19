#!/usr/bin/env python
# -*- coding: utf8 -*- 

import sys
import numpy as np

def sm_hist(data, delta=5, n_bin=None, range_=None):
	dataMin = np.floor(data.min())
	dataMax = np.ceil(data.max())
	n_bin = np.ceil(1.*(dataMax-dataMin) / delta)
	range_ = (dataMin, dataMin + n_bin * delta)
	counts, bin_edges = np.histogram(data, n_bin, range_, density = False)
	counts = np.hstack((np.array([0]), counts, np.array([0])))
	bin_edges = np.hstack((bin_edges[0], bin_edges))
	return counts, bin_edges

INDENT = 4
SPACE = " "
NEWLINE = "\n"

def to_json(o, level=0):
	"""
	Provided [here](http://stackoverflow.com/questions/10097477/python-json-array-newlines) 
	by [Jeff Terrace](http://jeffterrace.com/).
	Brunetto Ziosi added the sorted.
	"""
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
		if data is not None:
			self.idxs = data["idx"]
			self.data = data["data"]
	def stats(self):
		self.mean = self.data.mean()
		self.std = self.data.std()
		return self.mean, self.std
	def freqs(self, delta=None):
		if delta == None:
			self.delta = ((self.data.max()-self.data.min()) / 50.)
		else:
			self.delta = delta
		if self.delta == 0:
			print "Zero delta, exit"
			sys.exit(1)
		self.counts, self.bins = sm_hist(self.data, delta=self.delta)
		self.counts = self.counts / (1.*self.counts.sum()*self.delta)
	def percentile(self, which):
		return np.percentile(self.data, which)
	def minPercentile(self, which):
		# Wikipedia definition
		idx = int(np.ceil((which / 100.) * self.data.size))
		return np.sort(data["s"].data)[idx]



# Importare nell’ambiente grafico files i files di output di JAGS.
idxType = [("var", "|S30"), ("start","<i8"), ("stop", "<i8")]
dataType = [("idx", "<i8"), ("data", "<f8")]

idxs = np.genfromtxt("CODAindex.txt", dtype=idxType)
dataAll = np.genfromtxt("CODAchain1.txt", dtype=dataType)

data = {}

for var in idxs:
	if data.has_key(var[0]):
		print var[0], " duplicated! Exit!"
		sys.exit(1)
	data[var[0]] = dataO(var[0], dataAll[var["start"]-1:var["stop"]])




for key in data.keys():
	# Calcolare la media e deviazione standard di una sequenza di valori (p.e. s del file allegato
	# ha media 28.55 e standard deviation 16.5)
	print key, data[key].stats()
	
	# Calcolare l’intervallo pi ́u corto che racchiude l’x % dei valori (p.e. s del file allegato ha
	# un intervallo al 95 % che va da 0 a 56)
	for key in data.keys():
		data[key].freqs()


#• a) importare nell’ambiente grafico files i files di output di JAGS. Prendete come esempio
#quelli all’URL:
#http://www.brera.mi.astro.it/∼andreon/corso metodi bayesiani/CODAindex.txt
#http://www.brera.mi.astro.it/∼andreon/corso metodi bayesiani/CODAchain1.txt
#CODAindex.txt indica cosa contiene CODAchain.txt (pi ́
 #u variabili, in coda una dietro
#l’altra) e dove queste iniziano e finiscono. Per esempio s inizia alla riga 1 e finisce alla 50000
#del file CODAchain, mentre la variabile bkg inizia alla riga 50001 e finisce alla 100000. Si
#preveda gi`
 #a una certa flessibilit`a, il prossimo CODAindex.txt conterr`a un numero diverso
#di righe (p.e. 30).
#• b) calcolare la media e deviazione standard di una sequenza di valori (p.e. s del file allegato
#ha media 28.55 e standard deviation 16.5)
#• c) calcolare l’intervallo pi ́
 #u corto che racchiude l’x % dei valori (p.e. s del file allegato ha
#un intervallo al 95 % che va da 0 a 56)
#• d) plottare l’andamento di una variabile con il suo indice di sequenza (trace plot), per una
#variabile sola, e 8 alla volta. Si veda la fig 1, pannello di sinistra. Suggerimento: basta
#leggere la prima e la seconda colonna, e plottare una contro l’altra.
#• e) plottare la distribuzione delle frequenze di una variable (p.e. si veda il pannello di destra
#della Fig. 1 per la variabile s). Si noti che:
#– l’integrale della distribuzione  ́e uno per definizione. Suggerimento: farsi l’integrale a
#occhio sulla figura, se non torna meglio verificare.
#– la forma della distribuzione  ́e indipendente dal bin size (o kernel) usato, per bin sizes
#ragionevoli. Suggerimento: se cambiando il bin size, la distribuzione si muove ...
#Huston, c’ ́e un problema.
#In SuperMongo basta usare il comando histogram e normalizzare, per esempio:
#set myhisto=histogram(intr.scat:mycent)/step/dimen(intr.scat).
#• f) plottare il classico grafico con i contorni di confidenza per due parametri (si usi come
#esempio s e bkg). I contorni devono essere di forma smooth e non fissata (no a ellissi o
#cerchi a priori. Prevedete gi`
 #a che vi possano essere due isole), e devono includere il 68
#% e il 95 % dei punti. Non  ́e necessario che il conto sia esatto,  ́e accettabile una certa
#approssimazione, se la curva che dovrebbe contenere il 68 % dei punti ne contiene invece
#il 65 %, amen.
#Si noti che non  ́e possibile settare una threshold pari al massimo - ”magic numer” (come
#suggerito da Avni 1976, o Numerical Recipes), perch ́e questa contiene la percentuale voluta
#dei punti solo nel caso gaussiano (pi ́
 #u un certo numero di condizioni, non vi tedio). Va
#invece trovato il contorno che racchiude la percentuale voluta dei punti.
#In SuperMongo si ottiene per esempio creando una matrice/immagine (io uso 33x33 pixels)
#in cui ogni pixel ha un valore pari al numero di punti cascano nell’area del pixel. Usando
#il comando contour dopo aver settato i livelli si ottiene il plot in Fig 2. I livelli sono tali
#per cui per cui la somma dei valori dei pixels con valore maggiore del livello  ́e pari al 68
#% del numero totale di punti. I maghi tra voi potranno anche tener conto dell’effetto di
#smoothing indotto dalla pixellizzazione, mentre gli altri possono ignorare la complicazione.

#• g) nella prima lezione del corso calcoleremo i contorni di confidenza su un parametro
#secondo la ricetta frequentista. A tale scopo, lo studente legga la matrice
#http://www.brera.inaf.it/utenti/andreon/JKCS041 nH T.grid
#e si prepari un programmino che calcola, e plotta, somme o max lungo le righe e/o colonne
#e che plotti la matrice (ispiratevi alla Fig 2). La matrice  ́e scritta nel formato xi , yi, valore.
















out_file = open("bplots.json","w")
out_file.write(to_json(dataOut))
out_file.flush()
out_file.close()



