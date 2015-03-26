2015-03-25
==========

Slides:
user: statistica
pwd: bayesiana

# Poisson    

```
model {
 obsn ~ dpois(s)
 s ~ dunif(0,1.0e7)
}
```

conditional independent data:

```
model {
for ( i in 
  1:length(obsn) ) {
obsn[i] ~ dpois(s)
}
s ~ dunif(0,1.0e7)
}
```


# Binomial    

```
model {
obsn ~ dbin(f,n)
f ~ dbeta(1,1)
}
```

Astro case: dry merger rage (De Propis et al. 2010)    
Observed 2 mergers (obsn = 2) in a sample of n = 2127
What if obsn=0


# Prior

essential part of Bayes theorem    
if you decided to ignore it you are goind wrong with your results    

(Malmquist-xxx bias)    

p(theta|data) = c * p(data|theta) * p(theta)

```
model {
obsn ~ dpois(s)
s <- pow(tmps, -0.666666666666) # euclidian counts
tmps ~ dunif(0,10000)
}

obsn <- 4
```

It is not needed to have the perfect shape of the prior, you only need
to be close to it.    

In JAGS, truncate distrubution for positive numbers: T(0,)   

# source + background

```
model {
obstot ~ dpois(s+bkg/C)
obsbkg ~ dpois(bkg)
s ~ dunif(0, 1.0e7)
bkg ~ dunif(0, 1.0e7)
}
```

data with different bkg values:   

```
model {
  for (i in 1:length(obstot)) {
    obstot[i] ~ dpois(s+bkg[i]/C[i])
    obsbkg[i] ~ dpois(bkg[i])
    bkg[i] ~ dunif(0, 1.0e7)
s ~ dunif(0, 1.0e7)
}
```





