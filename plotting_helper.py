#import libraries 

import numpy as np
#from numba_stats import norm
from scipy import stats
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab
import math
import seaborn as sns
from iminuit import cost, Minuit 

#define function for fitting gaussian photopeaks with fitting tool iminuit

# @njit
def threshold(x, x0, sigma, Eth):
    # return (1+stats.norm.cdf(x+Eth, x0, sigma))
    return stats.norm.cdf(x+Eth, x0, sigma)

# @njit
def shelf(x, x0, sigma):
    return (1.-norm.cdf(x, x0, sigma))

def shelf_scipy(x, x0, sigma):
    return (1.-stats.norm.cdf(x, x0, sigma))

# @njit
def exp_tail(x, x0, sigma, gamma):
    return (np.exp(gamma*(x-x0)))

# @njit
def linear_tail(x, x0, sigma, m):
    return (1+m*(x-x0))

# @njit
def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))

def gauss_plus_tail_depth(x, A, BoverA, x0, sigma_gauss, gamma, CoverB, linearm, sigma_ratio, background):
    return (A*gaussian(x,x0,sigma_gauss) + A*BoverA*exp_tail(x, x0, sigma_gauss/sigma_ratio, gamma)*shelf(x,x0, sigma_gauss/sigma_ratio) + \
            A*BoverA*CoverB*linear_tail(x, x0, sigma_gauss/sigma_ratio, linearm)*shelf(x,x0, sigma_gauss/sigma_ratio) + \
            background*shelf(x,x0, sigma_gauss/sigma_ratio))

def gauss_plus_tail_pdf(x, BoverA, x0, sigma_gauss, gamma, CoverB, linearm, sigma_ratio, background, Emin = 53., Emax = 75.):
    return gauss_plus_tail_depth(x, 1., BoverA, x0, sigma_gauss, gamma, CoverB, linearm, sigma_ratio,  background)/\
    quad(gauss_plus_tail_depth, Emin, Emax, args=(1., BoverA, x0, sigma_gauss, gamma, CoverB, linearm, sigma_ratio, background))[0]

def gauss_minus_linear_tail_depth(x, A, BoverA, x0, sigma_gauss, gamma, CoverB, linearm, sigma_ratio , background):
    return (A*gaussian(x,x0,sigma_gauss) + A*BoverA*exp_tail(x, x0, sigma_gauss/sigma_ratio, gamma)*shelf(x,x0, sigma_gauss/sigma_ratio) + \
            background*shelf(x,x0, sigma_gauss/sigma_ratio))
            
def gauss_minus_linear_tail_pdf(x, BoverA, x0, sigma_gauss, gamma, CoverB, linearm, sigma_ratio, background, Emin = 640., Emax = 672.):
    return gauss_minus_linear_tail_depth(x, 1., BoverA, x0, sigma_gauss, gamma, CoverB, linearm, sigma_ratio, background)/\
                                        quad(gauss_minus_linear_tail_depth, Emin, Emax, args=(1., BoverA, x0, sigma_gauss, gamma, CoverB, linearm, sigma_ratio, background))[0]




#defines a plotting function for gaussian photopeaks
#using iminuit plotting function (operated by CERN's ROOT team), calculates an optimal fit according to a chi-squared cost function 
#returns parameters and plot of fit
#note: will not work effectively on non-gaussian distributions

#energy is in kev
#fit functions are defined above 
#title refers to the dataset name


def minuit_plot(raw_energies, min_energy, max_energy, peak_energy, fit_func_depth, fit_func_pdf, title):  
    

    #make energy cuts to make fitting more accurate
    energies=[]
    for energy in raw_energies:
        if energy < max_energy and energy > min_energy: #these cuts can be changed
            energies.append(energy)
            
        
    #define plotting axes before deciding if youre gonna do a minuit fit 
    fig, axs = plt.subplots(1, 1, figsize=(12,12 ))

    

    #plot a histogram of energies 
    hist, bin_edges = np.histogram(energies, bins=100)
    hist=hist/sum(hist)


    ##################### SETTING UP MINIMIZATION #######################################################################################

    #define chi squared or 'cost' function of fit to data
    c = cost.UnbinnedNLL(energies, fit_func_pdf)


    n, bins, patches = axs.hist(energies,histtype="step", bins=100, color ='cornflowerblue') #no range bc we cut energies when we read them in
    peak = bins[np.where(n == n.max())][0]  #define peak

    #define expected bounds and center of peak
    spectral_line = peak_energy
    peakMin=min_energy
    peakMax=max_energy                     


    ### Set up the minimization procedure. The initialized values are from Steve's simulations
    global_gamma = 0.50
    global_CoverB = 0.13
    global_D = 0.028
    global_sigma_ratio = 0.85


    ## Minuit as a fitting function which uses a covariance matrix to calculate and minimize errors on a fit
    m = Minuit(c, BoverA=0.5, x0=peak_energy, sigma_gauss=0.8, gamma=global_gamma,background=0.,  CoverB=global_CoverB, linearm=global_D,  sigma_ratio=global_sigma_ratio,  Emin=peakMin, Emax=peakMax)
    m.limits[ "BoverA", "sigma_ratio","sigma_gauss", "gamma", "CoverB","linearm"] = (0, None)
    m.limits["x0"] = (peakMin, peakMax) 
    m.fixed["Emin", "Emax" , "background"] = True
    m.migrad()
    m.hesse() # another minuit method for calcualting errors

    #print(m.parCss)
    #print(m.errordef)
    #m.minos()

    ### Print out the FWHM of the Gaussian component calculated by minuit w errors
    print('Minuit FWHM = ', 2.355*m.values['sigma_gauss']*spectral_line/m.values['x0'], ' +/- ', 2.355*m.errors['sigma_gauss']*spectral_line/m.values['x0'])


    ################  Plotting ###############################################


    #define histogram for histogram and minuit fit 
    hist,binedges,_ = axs.hist(energies, histtype="step",bins=100)
    bin_centers = np.array((binedges[:-1] + binedges[1:]) / 2)

    #define parCss from fit in terms of data
    popt = m.values[:-2]
    A = np.sum(hist)*(bin_centers[1]-bin_centers[0])/\
        quad(fit_func_depth, np.min(energies), np.max(energies), args = (1.0, *popt))[0]
    B = A*m.values['BoverA']
    C = B*m.values['CoverB']

    #define FWHM and errors
    FWHM = 2.355*m.values['sigma_gauss']*spectral_line/m.values['x0']
    FWHM_errors = 2.355*m.errors['sigma_gauss']*spectral_line/m.values['x0']

    #plot combined fit 
    axs.plot(bin_centers,fit_func_depth(bin_centers, A, *popt),color= 'red', lw=0.5)

    #plot gaussian fit on sCse plot
    axs.plot(bin_centers, A*gaussian(bin_centers, m.values['x0'], m.values['sigma_gauss']),color= 'red', ls='--', lw=0.5)

    #plot exponential tail on sCse plot
    axs.plot(bin_centers, B*exp_tail(bin_centers, m.values['x0'], m.values['sigma_gauss']/m.values['sigma_ratio'], gamma=m.values['gamma'])*shelf(bin_centers, m.values['x0'], m.values['sigma_gauss']/m.values['sigma_ratio']),color= 'red', ls='--', lw=0.5)


    #plt.plot(bin_centers, C*linear_tail(bin_centers, m.values['x0'], m.values['sigma_gauss']/m.values['sigma_ratio'], m.values['linearm'])*shelf(bin_centers, m.values['x0'], m.values['sigma_gauss']/m.values['sigma_ratio']),color= 'red', ls='--', lw=0.5)
    #plt.plot(bin_centers, C*step(bin_centers, m.values['x0'], m.values['sigma_gauss']/m.values['sigma_ratio'], m.values['linearm'])*shelf(bin_centers, m.values['x0'], m.values['sigma_gauss']/m.values['sigma_ratio']),color= 'red', ls='--', lw=0.5)
    #plot background 
    #print("C=", C)
    #print (step(bin_centers, m.values['x0'], m.values['sigma_gauss']/m.values['sigma_ratio'], m.values['linearm'])*shelf(bin_centers, m.values['x0'], m.values['sigma_gauss']/m.values['sigma_ratio']))

    axs.set_xlabel('Energy (keV)', fontsize=24.)
    axs.set_ylabel('Counts', fontsize=24.)


    #list the Minuit FWHM w errors on the plot
    axs.text(peakMin, fit_func_depth(bin_centers, A, *popt).max()*0.7, 'FWHM = ' + str(round(FWHM, 4))+ ' +/- ' + str(round(FWHM_errors, 4)) +' keV', fontsize=23.)
    axs.text(peakMin, fit_func_depth(bin_centers, A, *popt).max()*0.9, '$\mu$ = ' + str(round(m.values['x0'], 5)), fontsize=24.)
    axs.text(peakMin, fit_func_depth(bin_centers, A, *popt).max()*0.8,  "$\sigma$ = " + str(round(m.values['sigma_gauss']*spectral_line/m.values['x0'], 4)), fontsize=24.)
    axs.set_title(title, fontsize=26.)

    
    plt.show()
    
    







    #define spline function that:
# (1) takes a spline fit of a dataset
# (2) calculates a FWHM and FWTM from the fit
# (3) returns plots w spline fit and FWHM and FWTM listed and plotted 
# (4) returns ratio of FWHM/FWTM 


#NOTES:
#energy is in kev
#title refers to the dataset name

def spline(raw_energies, min_energy, max_energy, title):
    
    #make energy cuts to make fitting more accurate
    energies=[]
    for energy in raw_energies:
        if energy < max_energy and energy > min_energy: #these cuts can be changed
            energies.append(energy)

     #define plotting axes before deciding if youre gonna do a minuit fit 
    fig, axs = plt.subplots(1, 1, figsize=(12,12 ))
    
    bin_num = int(max_energy - min_energy)*4  
    
    #define histogram
    hist,binedges,_  = plt.hist(energies, histtype="step", bins=bin_num,weights=np.full_like(energies, 0.5), color ='cornflowerblue')

    #use axes from plotting function to plot spline fot from this function 
    
    #plt.yscale('log')
    axs.set_xlabel('Energy (keV)', fontsize=24.)
    axs.set_ylabel('Counts', fontsize=24.)
    axs.set_title("Spline Fit for " +str(title), fontsize= 26.)
           #4 bins per keV
    

    #plot a spline for the hist 
    bin_centers = np.array((binedges[:-1] + binedges[1:]) / 2)
    n, bins, patches = axs.hist(energies,histtype="step", bins=bin_num, weights=np.full_like(energies, 0.5), color ='cornflowerblue')
    spline = UnivariateSpline(bin_centers, n)
    axs.plot(bin_centers, spline (bin_centers), color = 'red',linestyle ='--', label='spline')

    #plot FWHM and FWTM as horizontal lines
    tenth_max = np.max(hist)/10
    half_max= np.max(hist)/2
    #axs.axhline(y = tenth_max, color = 'mediumorchid', linestyle = ':', label="FWTM") 
    axs.axhline(y = half_max, color = 'darkblue', linestyle = ':', label="FWHM") 

    
    #define FWHM and FWTM splines by subtracting a portion of the peak value from the hist values
    FWHM_spline = UnivariateSpline(binedges[:-1], hist-0.5*np.max(hist))
    FWTM_spline = UnivariateSpline(binedges[:-1], hist-0.1*np.max(hist))

    #calculate FWHM and FWTM by subtracting roots of y-shifted spline fit 
    FWHM = FWHM_spline.roots()[-1]-FWHM_spline.roots()[0]
    FWTM = FWTM_spline.roots()[-1]-FWTM_spline.roots()[0]

    #define errors 

    #integrate for total counts

    integral_spline = spline.integral(bin_centers[0], bin_centers[-1])
    print(f"Integral (UnivariateSpline): {integral_spline:.4f}")

    #write FWHM and FWTM on plot
    axs.text(min_energy+1, np.max(hist)*0.95,  "FWHM = " + str(round(FWHM, 2)) +" keV", fontsize =26.)
    axs.text(min_energy+1, np.max(hist)*0.85,  "FWTM = " + str(round(FWTM, 2)) +" keV", fontsize =26.)
    axs.text(min_energy+1, np.max(hist)*0.75,  "FWTM/FWHM = " + str(round(FWTM/FWHM, 6)) +" keV", fontsize =26.)

    axs.legend(fontsize=16.)
    plt.show()
 #plot a spline fit of the dataset and calculate FWHM and FWTM 
    return FWHM, FWTM

#MULTI SPLINE
#plot photopeaks across anneals for Cs-137 by scaling by counts/s 

#energies is an list of energy lists for each dataset of DC or AC strips in keV
#min_energy is lower bound in keV
#max_energy is upper bound in keV
#change colors from defult color w color list
#exposure time list is a list of total exposure time in mins for each data set 
#label list is a list of strings for labels in the legend


def multi_spline(energies, min_energy, max_energy, min_count, max_count,  exposure_time_list, label_list, colors, colordefault=True, log=False, Q_factor=False, vline=False): 
    
    #import libraries needed for this function
    from scipy.interpolate import UnivariateSpline
    from scipy.signal import find_peaks
    
    if colordefault==True:
        colors = ["black",  "green", "yellow", "orange", "red","indigo", "blue","cyan", "purple","pink","magenta", "grey"]
 
    fig, axs = plt.subplots(1, 1, figsize=(24,12))

    #define list of bins for each hist s.t there are 4 bins per keV:
    bin_num = []
    for i in energies:
        bin = int((np.max(i)-np.min(i)))*4
        bin_num.append(bin)

    #empty list for tracking Q factor among anneals
    Q_list=[]

    for i in range(len(energies)):
    
        #define histogram
        hist,binedges,_  = plt.hist(energies[i], histtype="step", bins=bin_num[i],weights=np.full_like(energies[i], 0.5/exposure_time_list[i]), color=colors[i])
    
        #plot a spline for the hist 
        bin_centers = np.array((binedges[:-1] + binedges[1:]) / 2)
        n, bins, patches = axs.hist(energies[i],histtype="step",  bins=bin_num[i], weights=np.full_like(energies[i], 0.5/exposure_time_list[i]), label =label_list[i], color=colors[i])
        spline = UnivariateSpline(bin_centers, n)
        
        if Q_factor== True:
            #find Q factor:  Q = (662-second peak)/height of second peak

            #find the second peak by considering just the lower half of the hist
            condition = bin_centers < 655.0
            lowbins = bin_centers[condition]
            lowcounts = n[condition]
            
            # Find the maximum counts from this range
            second_max = np.max(lowcounts)

            #now find the location of the max counts
            second_max_index = np.where(n == second_max)[0][0]  
            second_peak = bin_centers[second_max_index]
        
            Q = (661.7 - second_peak)/second_max
            print("Q-factor for", label_list[i], Q)
            Q_list.append(Q)
        
                
        #integrate for total counts
    
        integral_spline = spline.integral(bin_centers[0], bin_centers[-1])
        #print(f"Integral (UnivariateSpline): {integral_spline:.4f}")
        #print(label_array[i])
    
    
    #for log plots if you want to 
    if log == True:
        plt.yscale('log')
    
    #label axes
    axs.set_xlabel('Energy (keV)', fontsize=26.)
    axs.set_ylabel('Counts $min^{-1}$', fontsize=26.)
    
    if vline==True:
    	plt.axvline(661.7, ls ="--", color = '#ff7f0e', lw=3) 


    axs.legend(fontsize=28.0, loc='upper left')
    plt.grid()
    axs.tick_params(labelsize=24.)
    axs.set_xlim(min_energy, max_energy)
    axs.set_ylim(min_count, max_count)
    
    plt.show()
    
    if Q_factor==True:
        return Q_list



 #fun lil function for characterizing trapping length for AC and DC sides 
#read in the spline plotting function which calculates and outputs the FWHM

def trapping_len(preRadSpline, postRadSpline): 
    #define params 
    d = 1.5 # distance between strips for COSI detectors in cm 
    FWHM_prerad= preRadSpline[0]
    FWHM_postrad = postRadSpline[0]
    delta_FWHM = np.sqrt(FWHM_postrad**2 - FWHM_prerad**2)
    
    lambda_trap = 0.68* d /(delta_FWHM/662) #0.68 is a conversion factor derived in litterature, refer to Steve
    
    return lambda_trap


def strip_sort(energies, stripID):
    from collections import defaultdict
    mapping = defaultdict(list)
    # Populate the mapping
    for strip, e in zip(stripID, energies):
        mapping[strip].append(e)
    
    # Ensure all numbers 1-37 are included in the final result, even if they have no corresponding values
    all_strips = list(range(1, 38))
    grouped_energies = [mapping[strip] for strip in all_strips]
    
    #print(len(grouped_energies))
    
    #for i in range(len(grouped_energies)):
    #    spectra(grouped_energies[i], i+1)
    #print(len(grouped_energies))
    return grouped_energies




#function which gives the spectra of individual strips and plots them
# takes in a list of energies and uses strip_sort function to sort them by strip
#energies should be energies for all events 
#stripID is list of strip IDs that are also read in for all events

def strip_spectra(energies, stripID):
    
    strip_energies=strip_sort(energies, stripID)  
    for i in range(len(strip_energies)):
        
    
        bin_num = 300#int(np.max(energies) - np.min(energies))*1 
        plt.hist(strip_energies[i], bin_num, histtype="step")

        plt.title(  "strip "+ str(i+1))
        
        #vertical lines to mark expected decays to see where strip by strip gain shifts might be
        #plt.axvline(661.7, color="purple", ls = "--")
        #plt.axvline(653, color="r", ls = "--")
        plt.xlim(500, 700)
        plt.legend()
        plt.show()



# Define the exponential decay function
def exponential_decay(t, A, tau):
    return A * np.exp(-np.absolute(t) / tau) 
    
def double_exponential(x, A, B, C, D):
    return C * np.exp(- np.absolute(A * x)) + D* np.exp(- np.absolute(B * x))


#fits a double exponential to dataset and plots each exponential decay, along with the double decay on top of the datapoints 
#used the functions exponential_decay() and double_exponential defined above 
#initial guess is list: [1/tau1, 1/tau2, Amplitude1, Amplitude2]
#option to calculate chi-squared of fit 

def fit_and_plot_double_exponential(x_data, y_data, guess, title, chisquared=False):
    from scipy.optimize import curve_fit
    from scipy.stats import chisquare

    #bounds for parameters
    bounds = ([0, 0, 19,110], [np.inf, np.inf,22,150 ])
    
    # Initial guess for the parameters
    initial_guess = guess
    
    # Fit the double exponential function to the data
    params, covariance = curve_fit(double_exponential, x_data, y_data, p0=initial_guess, bounds=bounds)
    #print(params, covariance)
    
    # Extract the fitted parameters}
    A, B, C, D  = params
    #print(params)
    
    # Generate fitted values
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = double_exponential(x_fit, *params)

    print("tau1 =", 1/params[0])
    print("tau2 =", 1/params[1])
    print("A1 =", params[2])
    print("A2 =", params[3])
    print("errors:" + str(np.sqrt(np.diag(covariance))))

    #plt.text(30, 80, "Time constant1" +str(1/params[1]))
    
    y_fit1 = exponential_decay(x_fit, params[2], 1/params[0])
    y_fit2 = exponential_decay(x_fit, params[3],   1/params[1])

   
    if chisquared == True:
        #calculate chi square of double exponential 
        y_est=[]
        for i in range(len(x_data)):
            y_est.append(double_exponential(x_data[i], *params))
            
        chi_square = chisquare(y_data, y_est)
        print("Chi Square:", chi_square)
    
    # Plot the original data and the fitted curve
    plt.figure(figsize=(24, 12))
    plt.scatter(x_data, y_data, color='blue', label='Data', lw =10)
    plt.plot(x_fit, y_fit, color='red', label='All Non-native Defects', lw=3)
    plt.plot(x_fit, y_fit1, color ='#ff7f0e',label="Primary Defects", lw=3)
    plt.plot(x_fit, y_fit2, color ='#1f77b4', label="Secondary Defects", lw=3)
    plt.ylabel("$FWHM_{d}$  (keV)", fontsize=26)
    plt.xlabel("Time Annealing at 80$^{\circ}$C (hours)", fontsize = 26)
    plt.tick_params(labelsize=24.)

    #plt.ylim(0,20)
    plt.title(title)
    plt.legend(fontsize=26)
    plt.grid(True)
    #plt.show()
    
    #return time constants for each exponential
    return params[0], params[1]


#these functions are the same as other double exponenential plotting function but you fix:
#the amplitude and time constant of one of the exponential decays 
def double_exponential_fixed(x, B, D):
    return 21.4 * np.exp(- np.absolute(1/2040 * x)) +  D * np.exp(- np.absolute(B * x)) #fixed amplitudes


def fit_and_plot_double_exponential_fixed(x_data, y_data, guess, title, chisquared=False):
    from scipy.optimize import curve_fit
    from scipy.stats import chisquare

    bounds = ([0, 0], [np.inf, np.inf])
    # Initial guess for the parameters
    initial_guess = guess
    
    # Fit the double exponential function to the data
    params, covariance = curve_fit(double_exponential_fixed, x_data, y_data, p0=initial_guess, bounds=bounds)
    #print(params, covariance)
    
    # Extract the fitted parameters
    B, D  = params
    #print(params)
    
    # Generate fitted values
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = double_exponential_fixed(x_fit, *params)

    print("tau2 =", 1/params[0])
    print("A2 =", params[1])
    #print("A1 =", params[2])
    #print("A2 =", params[3])
    print("errors:" + str(np.sqrt(np.diag(covariance))))

    #plt.text(30, 80, "Time constant1" +str(1/params[1]))
    
    y_fit1 = exponential_decay(x_fit, 21.4,  1000)
    y_fit2 = exponential_decay(x_fit, params[1],  1/params[0])

   
    if chisquared == True:
        #calculate chi square of double exponential 
        y_est=[]
        for i in range(len(x_data)):
            y_est.append(double_exponential_fixed(x_data[i], *params))
            
        chi_square = chisquare(y_data, y_est)
        print("Chi Square:", chi_square)
    
    # Plot the original data and the fitted curve
    #plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='blue', label='Data')
    plt.plot(x_fit, y_fit, color='red', label='Double exponential')
    plt.plot(x_fit, y_fit1, label="Primary Defects")
    plt.plot(x_fit, y_fit2, label="Secondary Defects")
    plt.ylabel("$FWHM_{rad}$  (keV)", fontsize=26)
    plt.xlabel("Time Annealing (hours)", fontsize = 26)
    plt.tick_params(labelsize=24.)
    #plt.ylim(0,20)
    plt.title(title)
    plt.legend(fontsize=26)
    plt.grid(True)
    #plt.show()
    return params[0], params[1]

