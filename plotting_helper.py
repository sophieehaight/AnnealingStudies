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

def spline(raw_energies, min_energy, max_energy, title, FWHMline=False):
    
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
    

    
    #define FWHM and FWTM splines by subtracting a portion of the peak value from the hist values
    FWHM_spline = UnivariateSpline(binedges[:-1], hist-0.5*np.max(hist))
    FWTM_spline = UnivariateSpline(binedges[:-1], hist-0.1*np.max(hist))

    #calculate FWHM and FWTM by subtracting roots of y-shifted spline fit 
    FWHM = FWHM_spline.roots()[-1]-FWHM_spline.roots()[0]
    FWTM = FWTM_spline.roots()[-1]-FWTM_spline.roots()[0]

    if FWHMline==True:
        axs.plot([FWHM_spline.roots()[0],FWHM_spline.roots()[-1] ], [half_max, half_max], ls="--", color='black', label ="FWHM")
    else:
        axs.axhline(y = half_max, color = 'darkblue', linestyle = ':', label="FWHM") 


    #define errors 

    #integrate for total counts

    integral_spline = spline.integral(bin_centers[0], bin_centers[-1])

    #calculate error for gaussian distribution 
    gaussian_error = FWHM/np.sqrt(integral_spline*2)
    print(f"Error on FWHM: {gaussian_error:.4f}")

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


def multi_spline(energies, min_energy, max_energy, min_count, max_count,  exposure_time_list, label_list, colors= ["black"]+ sns.color_palette("husl", 10),  log=False, Q_factor=False, vline=False, splineline=False): 
    
    #import libraries needed for this function
    from scipy.interpolate import UnivariateSpline
    from scipy.signal import find_peaks
    import seaborn as sns 
    

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
        if splineline==True:
            axs.plot(bin_centers, spline (bin_centers), color = 'red',linestyle ='--', label='spline')
        
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


########################################################################################################################################################################################

def errors(datasets, FWHMs):
    errors=[]
    for data, fwhm in zip(datasets, FWHMs):
        counts=len(data)
        err = fwhm/np.sqrt(8*counts)
        errors.append(err)
        
    return errors


###########################################################################################################################################################################################
# Modified mean_std function that calculates min_energy based on 5 consecutive bins with counts >= 10% of peak count
def mean_std(ax, energies,  title, max_energy=665., scale_factor=1.0, RT=False):
    
    
##########################to determine the min energy we will consider when counts fall below 10% of the peak height#################################

    #to do this we need to filter through the energies to retain only the energies between the peak and the backscatter:
    cutoff_energies = energies[(energies >= 550. ) & (energies <= 662)] 
    
    #define a histogram of these energies:
    bin_num = int((662 - 550) * 4)
    # Create the histogram to get bin counts and edges
    counts, bin_edges = np.histogram(cutoff_energies, bins=bin_num, range=(550, 662))
    
    
    # Find the peak count (maximum bin count)
    peak_count = np.max(counts)
    
    # Calculate 10% of the peak count
    threshold = 0.1 * peak_count

    # Find the first instance of 5 consecutive bins with counts <= 10% of peak count
    min_energy = 550  # this is a point we know will be before the backscatter
    
    consecutive_count = 0  # Counter for consecutive bins above the threshold
    
    for i in range(bin_num - 1, -1, -1): #cycle through bins from highest to lowest
        if counts[i] <= threshold:
            #print( counts[i], threshold, peak_count)
            consecutive_count += 1
        else:
            consecutive_count = 0  # Reset counter if we find a bin with insufficient counts
        
        # If we find 5 consecutive bins, set min_energy
        count_num=5
        if consecutive_count >= count_num:
            min_energy = bin_edges[i - count_num-1]  # The min_energy is the start of the bin sequence
            break
        
        if RT==True: # RT spectra merges with the backscatter so we manually set the lower bound
            min_energy = 475
############################## Calculate the mean, std dev, and FWHM ########################################################
    
    #now that we have the min energy we can cut the data:
    energies = energies[(energies >= min_energy) & (energies <= max_energy)]  
    
    # Set the number of bins for the histogram (adjust as needed)
    bin_num = int((max_energy - min_energy) * 4)
    
    mean = np.mean(energies)
    std_dev = np.std(energies)
    FWHM = 2.355 * std_dev
    
    # Scale the font size based on the scale_factor
    font_size = 10 * scale_factor
    label_size = 8 * scale_factor
    legend_size = 8 * scale_factor
    
    # Create the histogram on the given axis (ax)
    ax.hist(energies, bins=bin_num, range=(min_energy, max_energy), alpha=0.6, color='g', label=f'FWHM: {FWHM:.2f} keV')
    
    # Plot vertical lines for the mean and standard deviations
    ax.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    ax.axvline(mean - std_dev, color='b', linestyle='dashed', linewidth=2, label=f'Mean - 1 Std Dev: {mean - std_dev:.2f}')
    ax.axvline(mean + std_dev, color='b', linestyle='dashed', linewidth=2, label=f'Mean + 1 Std Dev: {mean + std_dev:.2f}')
    
    # Add labels and title with scaled font size
    ax.set_xlabel('Energy (keV)', fontsize=label_size)
    ax.set_ylabel('Counts', fontsize=label_size)
    ax.set_title(title, fontsize=font_size)
    
    # Adjust the legend font size and marker size
    ax.legend(fontsize=legend_size, markerscale=2*scale_factor)
    ax.tick_params(labelsize=font_size)
    
    return FWHM

############################################################################################################################################################################################



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




##############   functions for fitting and interpolations ##########################################

#exponential decay fititng function

 # Define the exponential decay function
def exponential_decay(product, A, l):
    return A * np.exp(-l / product) 

def exponential_decay2(t, A, tau, C):
    return A * np.exp(-np.absolute(t) / tau) + C  # Asymptote at a non-zero baseline

def decayfit(x_data, y_data, A_init, tau_init, C_value, xmin, xmax):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # Convert input data to numpy arrays
    x_array = np.array(x_data)  # Time points
    y_array = np.array(y_data)  # Corresponding values
    
    # Set bounds for A and tau only, leave C as a constant
    bounds = ([0, 0], [np.inf, np.inf])  # Bounds for A and tau, no bound for C
    
    # Input initial guess parameters for the exponential decay fit
    initial_guess = [A_init, tau_init]  # Only A and tau are varied

    # Fit the data to the exponential decay function (C is fixed)
    popt, pcov = curve_fit(lambda t, A, tau: exponential_decay2(t, A, tau, C_value), x_array, y_array, p0=initial_guess, bounds=bounds)

    # Extract fitted parameters
    A_fit, tau_fit = popt
    C_fit = C_value  # Set C to the constant value

    print("errors:" + str(np.sqrt(np.diag(pcov))))
    
    # Return continuous arrays for plotting the exponential curve
    x_continuous = np.linspace(xmin, xmax, 100)  # Continuous time points for plotting
    y_continuous = exponential_decay2(x_continuous, A_fit, tau_fit, C_fit)

    return x_continuous, y_continuous, tau_fit, A_fit, C_fit
    
def exp_plot(x_data, y_data, A_init, tau_init, C_value, xmin, xmax, ymin, ymax, xlabel, ylabel, labels=False):
    plt.figure(figsize=(8, 6))

    # Calculate the time constant (tau) and print it
    tau_constant = decayfit(x_data, y_data, A_init, tau_init, C_value, xmin, xmax)[2]
    A_constant = decayfit(x_data, y_data, A_init, tau_init, C_value, xmin, xmax)[3]
    print("Decay Constant:", -tau_constant)
    print("Amp:", A_constant)

    # Scatter plot of data points
    plt.scatter(x_data, y_data)
    
    if labels:
        # Labels for data points
        for i, label in enumerate(labels):
            plt.text(x_data[i] + 65, y_data[i] + 0.025, label, ha='right', fontsize=11)

    # Plot exponential fit 
    x_fit, y_fit, tau_fit, A_fit, C_fit = decayfit(x_data, y_data, A_init, tau_init, C_value, xmin, xmax)
    plt.plot(x_fit, y_fit, 'g-', label='Exponential Decay')

    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)

    plt.grid(True)
    plt.legend()
    plt.show()




    # define a function with plots two datasets on 2 y axes and the fitted curves
def exp_plot_doubleaxis(x_data, xmin, xmax,xlabel, y_data1,  A_init1, l_init1, ymin1, ymax1, ylabel1, y_data2, A_init2, l_init2, ymin2, ymax2, ylabel2,  labels=False):
    fig, ax2 = plt.subplots(figsize=(8, 6))
    ax1 = ax2.twinx()
    

    #for first axis:
    
    #scatter plot of data points
    ax1.scatter(x_data, y_data1, label="associated trapping products")
    if labels!= False:
        #labels for data points
        for i, label in enumerate(labels):
            ax1.text(x_data[i]+65, y_data1[i]+0.025, label, ha='right', fontsize=11)

    #plot exponential fit 
    ax1.plot(decayfit(x_data, y_data1, A_init1, l_init1, xmin, xmax)[0], decayfit(x_data, y_data1, A_init1, l_init1, xmin, xmax)[1],'g-', label='Exponential Decay')

    
    #plt.title("Cs-137 DC Strips -- Simulation Data", fontsize=14)
    ax1.set_ylabel(ylabel1, fontsize=12)
    ax1.set_xlabel(xlabel, fontsize = 12)
    ax1.set_ylim(ymin1, ymax1)

    #plt.yscale("log")
    
    #ax1.grid(True)
    #plt.legend
    #plt.show()
    
    # Calculate the time constant (tau) and print it 
    l_constant1 = decayfit(x_data, y_data1, A_init1, l_init1, xmin, xmax)[2]
    A_constant1 = decayfit(x_data, y_data1, A_init1, l_init1, xmin, xmax)[3]
    print("Decay Constant:", -l_constant1)
    print("Amp:", A_constant1)
    #print(decayfit(x_data, y_data, A_init, l_init, xmin, xmax)[3])
    
    #for second axis:
    
    #scatter plot of data points
    ax2.scatter(x_data, y_data2, marker="*", label="experimental data")
#     if labels!= False:
#         #labels for data points
#         for i, label in enumerate(labels):
#             ax2.text(x_data[i]+65, y_data2[i]+0.025, label, ha='right', fontsize=11)

    #plot exponential fit 
    #ax2.plot(decayfit(x_data, y_data2, A_init2, l_init2, xmin, xmax)[0], decayfit(x_data, y_data2, A_init2, l_init2, xmin, xmax)[1],'r', label='')

    
    #plt.title("Cs-137 DC Strips -- Simulation Data", fontsize=14)
    ax2.set_ylabel(ylabel2, fontsize=12)
    ax2.set_xlabel(xlabel, fontsize = 12)
    ax2.set_ylim(ymin2, ymax2)

    #plt.yscale("log")
    
    ax2.grid(True)
    #ax2.legend()
    #plt.show()
    
    # Calculate the time constant (tau) and print it 
    l_constant2 = decayfit(x_data, y_data2, A_init2, l_init2, xmin, xmax)[2]
    A_constant2 = decayfit(x_data, y_data2, A_init2, l_init2, xmin, xmax)[3]
    print("Decay Constant:", -l_constant2)
    print("Amp:", A_constant2)



###################################################################################################################################################################################

# Double exponential function with tau1 fixed
def double_exp_fixed_tau1(t, A1, A2, tau2, C=3):
    tau1 = 375.0  # Fix tau1 to a known value (e.g., 1.0)
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C

# Function to fit double exponential with fixed tau1 and plot the result
def fit_double_exponential_fixed_tau1(t_data, y_data, initial_guess, C=3):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    bounds=([0, 70, 0], [38, 115, 75])
    # Fit the data using curve_fit, where only A1, A2, and tau2 are fitted
    popt, pcov = curve_fit(double_exp_fixed_tau1, t_data, y_data, p0=initial_guess, bounds=bounds)
    
    # Extract the optimal parameters
    A1_fit, A2_fit, tau2_fit = popt
    tau1_fixed = 375.0  # Fixed tau1 value
    print(f"Fitted Parameters: A1 = {A1_fit}, A2 = {A2_fit}, tau2 = {tau2_fit}, tau1 (fixed) = {tau1_fixed}")
    
     # Calculate the standard deviations (errors) on the parameters
    perr = np.sqrt(np.diag(pcov))  # Standard deviations (errors)
    
    print("errors:" + str(np.sqrt(np.diag(pcov))))
    print(pcov)
    
#     print(f"Parameter Errors:")
#     print(f"Error on A1 = {perr[0]}")
#     print(f"Error on A2 = {perr[1]}")
#     print(f"Error on tau2 = {perr[2]}")
    
    # Generate fitted data for the whole curve
    t_fit = np.linspace(min(t_data), max(t_data), 1000)
    y_fit = double_exp_fixed_tau1(t_fit, *popt)
    
    # Generate the individual exponential components
    exp1 = A1_fit * np.exp(-t_fit / tau1_fixed)+ C  # First exponential component
    exp2 = A2_fit * np.exp(-t_fit / tau2_fit) + C  # Second exponential component
    
    # Plot the data, the fitted curve, and the individual exponentials
    plt.figure(figsize=(8, 6))
    
    # Plot the original data points
    plt.scatter(t_data, y_data, color='red', label='Data points')
    
    # Plot the full fitted curve
    plt.plot(t_fit, y_fit, label=f'Fitted curve', color='blue', linewidth=2)
    
    # Plot the individual exponential components
    plt.plot(t_fit, exp1, '--', label=f'Component 1: A1 * exp(-t/tau1)', color='green')
    plt.plot(t_fit, exp2, '--', label=f'Component 2: A2 * exp(-t/tau2)', color='orange')
    
    # Labels and legend
    plt.xlabel('Time (t)')
    plt.ylabel('Value (y)')
    plt.legend()
    plt.title('Double Exponential Fit with Individual Components')
    plt.grid(True)
    plt.show()

    return popt

    # Objective function for lmfit
def objective(params, t, data):
    A1 = params['A1']
    A2 = params['A2']
    tau2 = params['tau2']
    model = double_exp_fixed_tau1(t, A1, A2, tau2)
    return model - data

# Fit function using lmfit
def fit_with_lmfit(t_data, y_data, initial_guess, lambda_reg=0.1):
    import lmfit
    # Create a model object
    params = lmfit.Parameters()
    
     # Set bounds for A1 (0 <= A1 <= 38), A2 and tau2 with no bounds
    params.add('A1', value=initial_guess[0], min=0, max=38)  # Upper bound for A1
    params.add('A2', value=initial_guess[1], min=0)  # No bounds for A2
    params.add('tau2', value=initial_guess[2], min=0)  # No bounds for tau2


    # Create a Model object
    model = lmfit.Model(double_exp_fixed_tau1, independent_vars=['t'])
    model.set_param_hint('tau1', value=375.0, vary=False)  # Fix tau1

    # Perform the fit
    result = model.fit(y_data, params, t=t_data)
    print(result.fit_report())
    return result

#trying to figure out how to calculate effective FWHM 
def eff_FWHM(energies, min_energy, max_energy, title):
    
    bin_num = int(max_energy - min_energy)*4
    print(bin_num)
    hist, bins = np.histogram(energies, bins=bin_num, range=(min_energy, max_energy), density=True)

    # Step 2: Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(hist * np.diff(bins))  # Cumulative sum, multiplying by bin widths

    # Step 3: Find the two points corresponding to the 12% and 88% cumulative probability
    lower_bound = np.searchsorted(cdf, 0.12)  # First point where CDF reaches 12%
    upper_bound = np.searchsorted(cdf, 0.88)  # First point where CDF reaches 88%

    # Step 4: The actual points on the histogram corresponding to the two bounds
    left_point = bins[lower_bound]
    right_point = bins[upper_bound]

    # Step 5: Calculate the distance between the pointsd
    eff_FWHM = right_point - left_point

    # Output the result
    print(f"Left point: {left_point}")
    print(f"Right point: {right_point}")
    print(f"Distance between points: {eff_FWHM}")

    # Optionally, plot the histogram and the identified points
    plt.hist(energies, bins=bin_num,range=(min_energy, max_energy),  alpha=0.6, color='g', label=f'Effective FWHM: {eff_FWHM:.2f} keV')
    plt.axvline(left_point, color='r', linestyle='dashed', label=f'Left point: {left_point:.2f}')
    plt.axvline(right_point, color='b', linestyle='dashed', label=f'Right point: {right_point:.2f}')
    plt.xlim(min_energy, max_energy)
    plt.title(title)
    #plt.ylim(max_energy)
    plt.legend()
    plt.savefig(f'Eff_FWHM/{title}')
    plt.show()
    
    return eff_FWHM


    ################################################### plotting functions I was using for unoised simulated data ##############################################################

    #add the gaussian noise expected from the COSI detector response (sans trapping)
#
def add_noise(data_ac, data_dc, noise_ac, noise_dc): #data is an array of datasets, noise is FWHM in keV


    std_dc = noise_dc/2.335
    std_ac = noise_ac/2.335
    
    #empty lists to store noisy data
    dc_noised =[]
    ac_noised=[]
    
    for i in range(len(data_dc)):
        
        bin_num_dc = len(data_dc[i])
        bin_num_ac = len(data_ac[i])
        
        noise_dc = np.random.normal(loc=661.657, scale=std_dc, size=bin_num_dc)
        noise_ac = np.random.normal(loc=661.657, scale=std_ac, size=bin_num_ac)
    
        #add noise to data 
        dc_noised.append(data_dc[i]+noise_dc)
        ac_noised.append(data_ac[i]+noise_ac)

    return ac_noised, dc_noised, noise_ac, noise_dc





    ####################################################################
    #plot raw data, noise, and noised data 
#steve later produced data that was already noised, making this function useless

def plot_noised_data(data_ac, data_dc, noise_ac, noise_dc):

    ac_noised = add_noise(data_ac, data_dc, noise_ac, noise_dc)[0]
    dc_noised = add_noise(data_ac, data_dc, noise_ac, noise_dc)[1]
    noise_ac = add_noise(data_ac, data_dc, noise_ac, noise_dc)[2]
    noise_dc = add_noise(data_ac, data_dc, noise_ac, noise_dc)[3]
    

    for i in range(len(data_dc)):
        
        #dc
        plt.hist(dc_noised[i]- 661.657, bins=150, label="noised data")
        plt.hist(data_dc[i], bins=150,  histtype="step",label ="raw data")
        plt.hist(noise_dc, bins=150, histtype="step",label ="noise")
        plt.xlim(650, 665)
        plt.title("DC strips: $[n \sigma]^{-1} = $"+ str(product_val[i]))
        plt.legend()
        plt.show()
        
        #ac
        plt.hist(ac_noised[i]- 661.657, bins=150, label="noised data")
        plt.hist(data_ac[i], bins=150, histtype="step", label ="raw data")
        plt.hist(noise_ac, bins=150, histtype="step",label ="noise")
        plt.xlim(655, 665)
        plt.title("AC strips: $[n \sigma]^{-1} = $"+ str(product_val[i]))
        plt.legend()
        plt.show()
      



#to find the error on the FWHM added by the spline method,
#iterate the calculations/fitting process and find the error of the distribution

# FWHM_distrib = []


# from itertools import zip_longest

# for i in range(200):
#     FWHM =  plot_spline(ac_array_list,dc_array_list)
#     #print(FWHM)
#     FWHM_distrib.append(FWHM)

# combined_FWHM = [list(filter(None, x)) for x in zip_longest(*FWHM_distrib, fillvalue=None)]

# #print(combined_FWHM)


# #now we find the actual detector noise 

# #define FWHM for dc data since holes / hole collecting strips will experience the most trapping 
# FWHMs_dc = product_val_means

# #define trapping product values for plotting against FWHM
# product_val = [ 10, 20,50, 100, 200, 500, 1000, 2000]
# labels = [ "10", "20", "50", "100cm", "200cm", "500cm", "1000cm", "2000cm"]


# #plot FWHM against trapping products

# plt.scatter(product_val, FWHMs_dc, marker="*")

# labels = ["10cm", "20cm", "50cm", "100cm", "200cm", "500cm", "1000cm", "2000cm"]
# for i, label in enumerate(labels):
#     plt.text(product_val[i]-10, FWHMs_dc[i]+0.75, label, ha='right', fontsize=11)
# plt.title("Cs-137 FWHM for Hole-Collecting Strips", fontsize=16)
# plt.ylabel("FWHM (keV)", fontsize=12)
# plt.xlabel("$[n \sigma]^{-1} = $ (cm)", fontsize = 12)
# #plt.legend()
# plt.grid(True)
# #plt.xscale("log")
