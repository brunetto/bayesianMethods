model in poisson_model_2.bug
data in poisson_data_2.data
compile, nchains(3)
initialize
update 1000
monitor set s, thin(10)
monitor set bkg, thin(10)
update 10000
coda *
data to useddata
samplers to usedsamplers
