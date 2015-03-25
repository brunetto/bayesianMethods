model in poisson_model_1.bug
data in poisson_data_1.data
compile, nchains(3)
initialize
update 1000
monitor set s, thin(10)
monitor set bkg, thin(10)
update 10000
coda *
data to useddata
samplers to usedsamplers
