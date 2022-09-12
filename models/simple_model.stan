/*
A custom three level hierarchical model designed to 
create example trials in the H1/H2 domain
*/
data {
    // Prior meta data for the strata HMs
    int<lower=1> S;
    real<lower=0> n0;
    
    // Data data
    int<lower=0> N;
    int<lower=1, upper=S> strata[N]; // Strata indicator
    matrix[N, 4] X;                  // Covariate matrix that indicates the treatment a patient recieved
    vector[N] y;                     // Outcome
}

transformed data {
  // Prior meta data for domain HMs
  int<lower=1> K=4;                 // number of parameters
  vector[2] n0s;
  vector[2] sigma0s;

  // Prior on within group mean
  for (i in 1:2){
    n0s[i] = 1.0;
    sigma0s[i] = sqrt(1/n0s[i]);
  }
}

parameters {
  
  // Domain hierarchical parameters
  vector[2] mu;
  vector[K] xi;
  vector<lower=-4, upper=4>[2] log_tau;
  
  // Strata heirarchical parameters
  matrix[K, S] csi;
  vector<lower=-4, upper=4>[K] log_eta;
  
  // main effect parameters
  vector<lower=0>[S] sigma2;
  vector[S] alpha;
}

transformed parameters {
  
  vector[K] b; 
  matrix[K, S] delta;
  matrix[K, S] beta;
  vector<lower=0>[S] sigma = sqrt(sigma2);
  vector<lower=0>[K] eta = exp(log_eta);
  vector<lower=0>[2] tau = exp(log_tau);
  vector<lower=0>[K] phi = eta ./ (1.0 + eta);
  vector<lower=0>[2] psi = tau ./ (1.0 + tau);

  // Hierarchical priors for the domains
  b[1] = mu[1] + xi[1]*sqrt(1/tau[1]);
  b[2] = mu[1] + xi[2]*sqrt(1/tau[1]);
  b[3] = mu[2] + xi[3]*sqrt(1/tau[2]);
  b[4] = mu[2] + xi[4]*sqrt(1/tau[2]);

  // Effects within strata
  for (k in 1:K){
    for (s in 1:S){
       if (k<3){
         delta[k, s] = b[k] + sigma[s]*csi[k, s]*sqrt(1/eta[k]/tau[1]);
       } else {
         delta[k, s] = b[k] + sigma[s]*csi[k, s]*sqrt(1/eta[k]/tau[2]);
       }
      
    }
  }

}

model {
  // Hierarchical priors for domains
  xi ~ normal(0, 1);
  mu ~ normal(0, sigma0s);
  // log_tau ~ uniform(-3, 3);

  // Hierarchical strata for 
  // log_eta ~ uniform(-3, 3);
  for (k in 1:K){
    csi[k, ] ~ normal(0, 1);
  }
  
  // Main effect priors
  alpha ~ normal(0, sigma/0.1);
  sigma2 ~ scaled_inv_chi_square(1, 1);
  
    // Likelihood
  for (n in 1:N){
    y[n] ~ normal(alpha[strata[n]] + (X[n, ]*delta[,strata[n]]), sigma[strata[n]]);
  }
}
