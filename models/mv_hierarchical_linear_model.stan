data {
    // Meta data
    int<lower=1> K; // Number of arms
    int<lower=1> N; // Number of observations
    int<lower=1> P; // Number of people
    int<lower=1> S; // Number of strata
    
    // Data Data
    vector[N] y;                     // Outcome
    int<lower=1, upper=S> strata[N]; // Strata indicator

    // Patient-level indexing
    int<lower=1, upper=N> first[P];  // Indicates the first value for person p
    int<lower=1, upper=N> final[P];  // Indicates the last value for person p
    int<lower=0, upper=K> arm[P];  // Index of the arm (0 is control)

}

transformed data {
  int<lower=1, upper=K+1> arm_[P];
  vector[S] zeroes; // should be able to do zeros_array(S); but my pystan is old
  for (p in 1:P) arm_[p] = arm[p]+1; // Shift the index
  for (s in 1:S) zeroes[s] = 1;
}

parameters {
  // Main effects
  vector[S] alpha;           // Strata specific control rate

  // Strata treatment effect parameters
  vector[S] delta_[K];        // Strata x Treatment interaction
  vector[K] delta_mean;       // hierarchical mean
  vector<lower=-10, upper=10>[K] log_eta; // log hierarchical variances
  
  // Variance
  cholesky_factor_corr[S] L; 
  vector<lower=0>[S] sigma;
}

transformed parameters {
  cov_matrix[S] Sigma; 
  matrix[K+1, S] Mu; // Mean in strata s for arm k-1 
  vector[S] delta[K];                          // Strata x Treatment interaction
  vector<lower=0>[K] eta = exp(log_eta*0.5);   // hierarcical standard deviation


  // Hierarchical prior for delta
  for (k in 1:K)  delta[k] = sigma .* (delta_mean[k] + delta_[k]*eta[k]);
  
  // Hierarchical mean
  for (s in 1:S) {
    Mu[1, s] = alpha[s]; // Control Arms
    for (k in 1:K) Mu[k+1, s] = alpha[s] + delta[k, s]; // Treatment arms
  }

  // Sampling covariances
  Sigma = quad_form_diag(multiply_lower_tri_self_transpose(L), sigma); // Creates the covariance matrix

}

model {
  
  // Main effects
  alpha ~ multi_normal(zeroes, Sigma/0.1);

  delta_mean ~ normal(0, 1);
  for (k in 1:K) delta_[k] ~ normal(0, 1);
  
  // Covariance prior
  sigma ~ cauchy(0, 1.0);
  L ~ lkj_corr_cholesky(1.0);
  
  // Likelihood
  // first[p]:final[p] give the indices for the observations from patient P
  // strata[first[p]:final[p]] give the dimensions for those indices.
  for (p in 1:P){
    y[first[p]:final[p]] ~ multi_normal(
      Mu[arm_[p], strata[first[p]:final[p]]],     // 
      Sigma[strata[first[p]:final[p]], strata[first[p]:final[p]]]
    );
  }
}

generated quantities {
  vector<lower=0>[K] eta2_ = 1.0 ./ (eta .* eta);  // Precision of across strata borrowing
  vector<lower=0>[K] ess = eta2_ ./ (1.0 + eta2_); // ESS for across strata borrowing
}
