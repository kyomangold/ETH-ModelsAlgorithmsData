# ------------------------------------------------- #
#             MAD - Exercise Set 12                 #
#          Monte-Carlo Sampling Methods             #
#                                                   #
#                Questions 1 and 2                  #
# ------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# ------------------------------------------------- #
# -------------- Plotting functions --------------- #
# ------------------------------------------------- #

def build_histogram(samples, a, b, Nbins ):
    output = np.zeros(Nbins)
    bins = np.zeros(Nbins)
    dx = (b-a)/(Nbins)

    for i in range (len(samples)):
        ibin = ((samples[i]-a)/dx)
        if( ibin < 0 ):
            ibin = 0;
        if( ibin > (Nbins - 1) ):
            ibin = Nbins - 1
        ibin = int(ibin)
        output[ibin] = output[ibin] + 1

    for i in range(Nbins):
      bins[i] = a + (i + 0.5) * dx
    return output, bins

def plot_histogram(samples,a, b, nbins,name):

    values, bins = build_histogram(samples,a, b, nbins);

    n_bins = len(bins)
    dx = bins[1]-bins[0]
    scale = 1/(dx*np.sum(values))

    fig = plt.figure()
    plt.bar(bins,values*scale,label='Samples')

    Z, _ = integrate.quad(P,-10,10)
    x = np.linspace(-15,15,1000)
    plt.plot(x,P(x)/Z,c='r',lw='3',label='Distribution')

    plt.legend()
    plt.savefig(name +'.png')

# ------------------------------------------------- #
# --------------- Helper functions ---------------- #
# ------------------------------------------------- #

def V (x):
    # potential function                                          
    return (x-2)**2 * (x+2)**2 * (x-4)**2 * (x+4)**2

def P(x):
    # gibbs distribution                                          
    T = 2000          
    return np.exp(-V(x)/T)

def proposal_rnd(x ,sigma): 
    # proposal random number: being at x        
    return np.random.normal(x, sigma)   # (mean,std)

# ------------------------------------------------- #
# --------------------- Methods ------------------- #
# ------------------------------------------------- #

def rejection_sampling(a,b,Ns,nbins):
    samples = []
    for i in range(Ns):
      u = np.random.uniform(a,b);
      if(np.random.uniform(0,1) <= P(u)):
         samples.append(u)
    accrate = len(samples)/Ns;
    print( "Acceptance rate: ", accrate)

    mean = np.sum(samples)/len(samples)
    print(mean)
    sum=0.0
    for i in range(len(samples)):
      sum += (samples[i]-mean)**2
    v1 = sum / len(samples)

    print("Estimated mean and std: ", mean, " ", np.sqrt(v1))

    return samples

def MCMC(sigma,Ns,nbins):
    samples = [0];
    P_old = P(samples[0]);
    k=0;
    for i in range(Ns):
      proposal = proposal_rnd( samples[i-1], sigma)
      P_new = P( proposal )

      acceptance_ratio = P_new/P_old;

      if( np.random.uniform(0,1) < acceptance_ratio ):
         samples.append(proposal)
         P_old = P_new
         k = k + 1
      else:
         samples.append(samples[i-1]);

    accrate2 = (k)/ (Ns);
    print("Acceptance rate: " , accrate2 )

    mean = np.sum(samples) / (Ns);
    sum=0.0
    for i in range(len(samples)):
      sum += (samples[i]-mean)**2
    v2 = sum / (Ns);
    print("Estimated mean and std: ", mean, " ",np.sqrt(v2))

    return samples

if __name__ == "__main__":
    
    # Parameters
    nbins = 200;      # Bins in histogram
    a = -15           # Range
    b = 15            # Range
    Ns = 1000000      # Number of iterations in the sampling algorithms
    sigma = 4.0       # variance of the proposal pdf

    samples_RS = rejection_sampling(a,b,Ns,nbins)
    samples_MCMC = MCMC(sigma,Ns,nbins)
      
    plot_histogram(samples_RS,a, b, nbins,name='rejection_sampling')  
    plot_histogram(samples_MCMC,a, b, nbins,name='MCMC')  

   
