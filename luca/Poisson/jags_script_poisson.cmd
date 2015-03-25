model in poisson_model.bug
data in poisson_data.data
compile, nchains(3)
initialize
update 1000
monitor set s, thin(10)
update 10000
coda *
data to useddata
samplers to usedsamplers
