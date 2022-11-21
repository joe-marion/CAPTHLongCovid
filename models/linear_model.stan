data {
    // Prior meta data for the strata HMs
    
    // Meta data
    int<lower=1> K; // Number of arms
    int<lower=1> N; // Number of observations
    int<lower=1> S; // Number of strata
    
    // Data Data
    vector[N] y; // Outcome
    int<lower=1, upper=S> strata[N]; // Strata indicator
    int<lower=0, upper=K> narm[N]; // Index of arm for observation N (0=control)
}

transformed data {
  int<lower=1, upper=K+1> narm_[N];
  for (n in 1:N) narm_[n] = narm[n]+1; // Shift the index
}

parameters {
  // Main effects
  vector[S] alpha;           // Strata specific control rate
  vector[S] delta[K];        // Strata x Treatment interaction
  vector<lower=0>[S] sigma;  // Obervation variance
}

transformed parameters {
  matrix[K+1, S] Mu; // Mean in arm k-1 for strata S

  for (s in 1:S) {
    Mu[1, s] = alpha[s]; // Control Arms
    for (k in 1:K) Mu[k+1, s] = alpha[s] + delta[k, s]; // Treatment arms
  }
  
}

model {
  
  // Covariance prior
  sigma ~ cauchy(0, 1.0);
  
  // Main effects
  for (s in 1:S){
    alpha[s] ~ normal(0, sigma[s]/0.1);
    for (k in 1:K) delta[k, s] ~ normal(0, sigma[s]);
  }
  
  // Likelihood
  for (n in 1:N){
    y[n] ~ normal(Mu[narm_[n], strata[n]], sigma[strata[n]]);
  }
  
}

