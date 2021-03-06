# Bayesian methods

by Stefano Andreon

Material [here](http://www.brera.mi.astro.it/~andreon/) 

## First day

* Probability is in  [0,1]
* P = 0, P = 1
* The sum rule (marginal probability): sum the line and/or the columns
* p(x,y) = p(x|y)p(y) = p(y|x)p(x)

p(theta|data) = c * p(data|theta) * p(theta)
posterior = c * likelyhood * prior

likelyhood = instrument resolution
posterior = probability of finding a value for that parameter = error/uncertainty on the parameter
prior = quantify your expectation

I can use as prior the posterior of a previous experiment.

It is wrong to fix a parameter to study another, the right thing is to marginalize, 
i.e., to sum over all the other parameters.

Anythong else is WRONG, you can do everything with it. 
If you follow the rules, the results will be right.

### Bayesian inference

Everything is in the form af probability.  
You need:

* formulation of a problem
* formulation of a prior
* Bayes theorem on the observed data

When you fit an (x, y), why do you assume a linear relation?

### Example 1

Everything is gaussian.  

model: $p(x|y,s^2) = gauss(x|y,s^2) = ...$
experiment: put a top model

Usually the likelyhood and the posterior are basically the same (they are superimposed).
The reason is that the prior usually is very "flat".

### Example 2

In the case of the neutrino mass the likelyhood and the posterior are different.

$n^0 -> p^+ + e^- + nu_e$

(bad way, small quantity as difference of large ones)

* gaussian model for the likelyhood p(x|y,s^2) = exp..
* prior: masses are positively defined
* experiment: find -4\pm2

posterior = prior * likelyhood

### Bayesian approach

Data is the sure thing and several parameters are considered.
Bayesian is conditional on the data.

The alternative approach is conditional on the true value.
In this approach the observed data are just one realization of the many 
data I might have observed.

In Bayesian approach you assume your observation is complete.

### JAGS

* JAGS do everything for you using a sort of Monte Carlo algorithm.
* You don't have to write code
* Prior file, data file, script describing the files and what you want
* R format (and others)

```r
My_var <-(val1, val2, val3)
My_var <-(
		val1, 
		val2, 
		val3)

```
* model: logical relationship between stuffs

```
# comment, but is better to not use it
model
{

}
```
* `receiver <- value`: put `value` in the variable `receiver`
* `x~normal`: `x` is taken form a normal distribution
* tau = 1/sigma^2 for the normal distribution

```
$ cat myfile.bug

model
{
m ~ duniff(0,10)
obsm ~ dnorm(m, 0.25)
}

$ cat data.dat

obsm <- -4

$ cat my_script.cmd

model in myfile.bug
data in myfile.data
compile, nchains(3) 			# run the program 3 times to avoid stalling chains, infinity etc.
initialize 						# find where to star
update 1000 					# discard the first 10^3 steps to not depend on the starting point
monitor set myvar, thin(10) 	# I want to store myvar once every 10 step (in MC codes near values are correlated)
update 10000 					# sampling of the posterior
coda * 							# default output names
data to useddata				# for debugging purpose
samplers to usedsamplers		# 

jags < my_script.cmd
```
if the 3 chains are different you need to skip more than 1000 initial values

## Second day

### Homework

* trace plot
* change the prior
* change the discarded values
* overplot the analytical posterior
* check the 90% upper limit (should be ~1.7)

### Poissonian

```
model {
obsn ~ dpois(s)
s~ dunif(0, 1.0e+7)
}

#### Add data

model {
for (i in 
	1:length(obsn)) {
		obsn[i] ~ dpois(s)
	}
	s ~ dunif(0,1.0e+7)
}

FIXME: where does it put the two extimate?

### Binomial

model {
obsn ~ dbin(f, n)
f ~ dbeta(1,1)
}

### Prior

The prior is always important, but in Bayesian analysis is a piece, so you can't forget it.
Examples: masses are positive, ...

**Malmquist-Eddington bias**: is the equivalent of the prior with another name.

Prior as power law because nature produces few big stuffs and many small ones.

p(mu|4) = c * p(4|mu) p(mu)

p(mu) = mu^{-beta}

We find that if the prior is a power law, the most probable value for an observation of 
4 photons is 1.5!

```
model {
obss ~ spois(s)
s <- pow(tmps, -0.666666666666666666666) # euclidean counts, but power law is called Pareto distro
tmps ~ dunif(0, 10000)

obss <- 4
```

Prior is important but do not be too anxious!
What matters is the gross shape of the prior where likelyhood is > 0.
The precise shape does not matter.

```
model {
# likelyhood
obsn ~ dpois(s)
# Gauss prior
s ~ dnorm(0,0.01) T(0,) # truncate
# uniform prior
s ~ dunif(0,1.0e+7)
}

```

### Background

```
model {
	obstot ~ dpois(s+bkg/C) # C is the solid angle
	obsbkg ~ dpois(bkg)
	s ~ dunif(0,1.0e+2)
	bck ~ dunif(0,1.0e+2)
}
```

Run with: 

* s+bkg, bkg, C (solid angle)
* 31, 1, 1
* 60, 30, 1 <- facciamo questo
* 60, 1, 1/30 # like neutrinos where you can't observe the Background
* 60, 3000, 100 # ???
* 170, 172, 1

### Combine data with different backgrounds (different observations)

But how can I have the same counts for the source?

```
model {
	for (i in 1:length(obstot)) {
		obstot[i] ~ dpois (s+bkg[i] / C[i])
		obsbkg ~ dpois(bkg)
		...
		
	
	}



}

```

### Homework

* consider the case of 40 measurements: is the posterior
narrower than the case of 1 obs?







