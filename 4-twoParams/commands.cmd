model in model.bug
data in data.dat
compile, nchains(3)
initialize 
update 1000
monitor set s, thin(10)
monitor set bkg, thin(10)
update 10000
coda *
data to useddata

