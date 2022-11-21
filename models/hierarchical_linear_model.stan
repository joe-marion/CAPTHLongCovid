data {
    // Prior meta data for the strata HMs
    
    // Meta data
    int<lower=1> K; // Number of arms
    int<lower=1> N; // Number of observations
    int<lower=1> S; // Number of strata
    
    // Data Data
    vector[N] y;                     // Outcome
    int<lower=1, upper=S> strata[N]; // Strata indicator
    int<lower=0, upper=K> narm[N];  // Index of arm for observation N (0=control)
}

transformed data {
  int<lower=1, upper=K+1> narm_[N];
  for (n in 1:N) narm_[n] = narm[n]+1; // Shift the index
}

parameters {
  // Main effects
  vector[S] alpha;           // Strata specific control rate
  
  vector<lower=0>[S] sigma;  // Obervation variance
  
  // Strata sharing parameters
  vector[S] delta_[K];        // Strata x Treatment interaction
  vector[K] delta_mean;       // hierarchical mean
  vector<lower=-10, upper=10>[K] log_eta; // log hierarchical variances

}

transformed parameters {
  matrix[K+1, S] Mu;                           // Mean in arm k-1 for strata S
  vector[S] delta[K];                          // Strata x Treatment interaction
  vector<lower=0>[K] eta = exp(log_eta*0.5);   // hierarcical standard deviation

  // Hierarchical prior for delta
  for (k in 1:K)  delta[k] = sigma .* (delta_mean[k] + delta_[k]*eta[k]);
  
  // Look-up matrix for effects
  for (s in 1:S) {
    Mu[1, s] = alpha[s]; // Control arms
    for (k in 1:K) Mu[k+1, s] = alpha[s] + delta[k, s]; // Treatment arms
  }
  
}

model {
  
  // Covariance prior
  sigma ~ cauchy(0, 1.0);
  
  // Main effects
  for (s in 1:S)  alpha[s] ~ normal(0, sigma[s]/0.1);
  
  delta_mean ~ normal(0, 1);
  for (k in 1:K) delta_[k] ~ normal(0, 1);
  
  // Likelihood
  for (n in 1:N){
    y[n] ~ normal(Mu[narm_[n], strata[n]], sigma[strata[n]]);
  }
  
}

generated quantities {
  vector<lower=0>[K] eta2_ = 1.0 ./ (eta .* eta);  // Precision of across strata borrowing
  vector<lower=0>[K] ess = eta2_ ./ (1.0 + eta2_); // ESS for across strata borrowing
}

