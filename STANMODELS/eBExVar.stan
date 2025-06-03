data {  
  int<lower=1> NI; // length of indices array
  int<lower=1> NS; // number of sources
  int<lower=1> NC; // size of count, bkg and exposure arrays
  array[NI] int indices; // indeces for the start/end of LCs
  array[NC] int<lower=0> counts; // observed counts for object 
  vector[NC] time; // exposure time (including ECF and EEF) 
  vector[NC] bkg;
  vector[NS] lgf0;
 }

parameters {

   // mean of NEV lognormal distribution of the population
   real<lower=-4, upper=0> A; 

   // scatter of NEV lognormal distribution of the population	
   real<lower=0.0, upper=1.0> B; 

   // log10 of mean flux of each light curve 
   vector[NS] LGCR_MEAN; 

   vector[NS] raw_sigma; // helper parameter
   vector[NC] raw; // helper parameter
}

transformed parameters {

   // log10(NEV) of each light curve
   vector[NS] LGNEV; 

   // log10(count rate) of each light curve epoch
   vector[NC] LGCR; 

   // SIGMA parameter of log-normal distribtion
   // of the single epoch count rates. The mean
   // parameter of this distribution is  LGCR_MEAN
   vector[NS] SIGMA; 
			
   // NEV is drawn from a lognormal distribution with 
   // mean A and scatter B that describes the population
   // It uses help std_normal() parameter for efficiency
   LGNEV = raw_sigma * B + A;

   // transforming between LGNEV and SIGMA parameter
   // of the lognormal distribution for the single epoch count rates
   SIGMA = sqrt(log10(pow(10,LGNEV)+1)/2.302585093);

   // single epoch count rates based on lognomral distribution with scatter SIGMA 
   // and mean parameter LGCR_MEAN
   // It uses help std_normal() parameter for efficiency
   for (i in 1:NS){
	LGCR[indices[i]+1:indices[i+1]] = raw[indices[i]+1:indices[i+1]] * SIGMA[i] + LGCR_MEAN[i];
   } 


}

model{

  // count expectation value for each epoch
  vector[NC] lambda; 

  // priors
  A ~ uniform(-4, 0); 
  B ~ uniform(0, 1.0); 
  raw ~ std_normal();
  raw_sigma ~ std_normal();
  LGCR_MEAN ~ normal(lgf0, 0.2);
 
  lambda = pow( 10, LGCR ) .* time + bkg;
  counts ~ poisson(lambda);
	
}
