#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba_stats import norm
from scipy import stats
from scipy.integrate import quad
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline
import dat2CTD
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from iminuit import cost, Minuit
import seaborn as sns
from multiprocessing import Pool

sns.set_context('talk', font_scale=1.0)
sns.set_palette('colorblind')

### Line profile parameters determined via simulations.
global_gamma = 0.50
global_CoverB = 0.13
global_D = 0.028
global_sigma_ratio = 0.85

### Important calibration lines.
source_dict = {'Am241': 59.5409, 'Cs137': 661.657,'Co57': 122.06065, 'Ba133': 356.0129, 'Na22': 1274.537}
erange_dict = {'Cs137': [630., 670.], 'Ba133': [330., 362.], 'Na22': [1220., 1290.], 'Am241': [50., 64.]}

### Functions for producing line profiles.

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

def gauss_plus_tail_depth(x, A, BoverA, x0, sigma_gauss, gamma, CoverB, D, sigma_ratio, background):
    return (A*gaussian(x,x0,sigma_gauss) + A*BoverA*exp_tail(x, x0, sigma_gauss/sigma_ratio, gamma)*shelf(x,x0, sigma_gauss/sigma_ratio) + \
            A*BoverA*CoverB*linear_tail(x, x0, sigma_gauss/sigma_ratio, D)*shelf(x,x0, sigma_gauss/sigma_ratio) + \
            background*shelf(x,x0, sigma_gauss/sigma_ratio))

def gauss_plus_tail_pdf(x, BoverA, x0, sigma_gauss, gamma, CoverB, D, sigma_ratio, background, Emin = 53., Emax = 75.):
    return gauss_plus_tail_depth(x, 1., BoverA, x0, sigma_gauss, gamma, CoverB, D, sigma_ratio,  background)/\
    quad(gauss_plus_tail_depth, Emin, Emax, args=(1., BoverA, x0, sigma_gauss, gamma, CoverB, D, sigma_ratio, background))[0]

def gauss_minus_linear_tail_depth(x, A, BoverA, x0, sigma_gauss, gamma, CoverB, D, sigma_ratio , background):
    return (A*gaussian(x,x0,sigma_gauss) + A*BoverA*exp_tail(x, x0, sigma_gauss/sigma_ratio, gamma)*shelf(x,x0, sigma_gauss/sigma_ratio) + \
            background*shelf(x,x0, sigma_gauss/sigma_ratio))
            

def gauss_minus_linear_tail_pdf(x, BoverA, x0, sigma_gauss, gamma, CoverB, D, sigma_ratio, background, Emin=640., Emax=672.):
    norm, _ = quad(
        gauss_minus_linear_tail_depth, 
        Emin, 
        Emax, 
        args=(1.0, BoverA, x0, sigma_gauss, gamma, CoverB, D, sigma_ratio, background)
    )
    
    # Evaluate array x over scalar norm
    return gauss_minus_linear_tail_depth(x, 1.0, BoverA, x0, sigma_gauss, gamma, CoverB, D, sigma_ratio, background) / norm

### We need a normalized version in order to perform unbinned fitting.
def gauss_plus_tail_pdf(x, BoverA, x0, sigma_gauss, gamma, CoverB, D, sigma_ratio, Emin = 640., Emax = 672.):
    return gauss_plus_tail(x, BoverA, x0, sigma_gauss, gamma, CoverB, D, sigma_ratio)/\
    quad(gauss_plus_tail, Emin, Emax, args=(BoverA, x0, sigma_gauss, gamma, CoverB, D, sigma_ratio))[0]

def get_FWHM_FWTM(x, y, C=0.):
    ### Function for determining the FWHM and FWTM of count histograms. 
    ### Also returns an estimate for the 1-sigma error bars.
    spline = UnivariateSpline(x, y-C, k=4)
    spline_roots = spline.derivative().roots()
    # print(spline_roots)
    spline_max = spline_roots[np.argmin(np.abs(spline(spline_roots) - np.max(y)))]
    # print(spline_max)
    fwhm_roots = UnivariateSpline(x, y-0.5*spline(spline_max)).roots()
    fwtm_roots = UnivariateSpline(x, y-0.1*spline(spline_max)).roots()
    # plt.figure()
    # plt.plot(x, y)
    # plt.plot(x, spline(x))
    # plt.show()
    # plt.close()
    fwhm = fwhm_roots[-1]-fwhm_roots[0]
    fwtm = fwtm_roots[-1]-fwtm_roots[0]
    fwhm_err = fwhm/np.sqrt(2.*np.sum(y))
    fwtm_err = fwtm/np.sqrt(2.*np.sum(y))
    return fwhm, fwtm, fwhm_err, fwtm_err



class DepthCalibrator_Am241:
    def __init__(self, AC_param_file, DC_param_file, AC_sim_ev, DC_sim_ev, sim_file, savefile=None):
        ### take in the Gaussian means from AC- and DC- side illumination with Am241 and compare to simulations to return a CTD->depth mapping function.
        self.DC_sim_CTD = []
        self.AC_sim_CTD = []
        self.AC_params = np.array([[[0.0,0.0] for p in range(37)] for n in range(37)])
        self.DC_params = np.array([[[0.0,0.0] for p in range(37)] for n in range(37)])
        self.AC_noise = 0.0
        self.DC_noise = 0.0

        ### Read in the Gaussian centroids determined for AC side illumination using dat2CTD.py
        with open(AC_param_file) as file:
            for line in file:
                if '#' not in line:
                    splitline = line.split(', ')
                    p = int(splitline[0])
                    n = int(splitline[1])
                    self.AC_params[p][n] = [float(splitline[2]), float(splitline[3])]

        ### Read in the Gaussian centroids determined for DC side illumination using dat2CTD.py
        with open(DC_param_file) as file:
            for line in file:
                if '#' not in line:
                    splitline = line.split(', ')
                    p = int(splitline[0])
                    n = int(splitline[1])
                    self.DC_params[p][n] = [float(splitline[2]), float(splitline[3])]


        with open(AC_sim_ev) as file:
            for line in file:
                self.AC_sim_CTD.append(float(line.split(',')[1]))
        self.AC_sim_CTD = np.array(self.AC_sim_CTD) 

        with open(DC_sim_ev) as file:
            for line in file:
                self.DC_sim_CTD.append(float(line.split(',')[1]))
        self.DC_sim_CTD = np.array(self.DC_sim_CTD)

        ### For each pixel, make histograms of the simulated CTD adding successively more noise
        ### Fill in the AC_params and DC_params accordingly
        noise_mu_sigma_AC = []
        for i in range(100):
            noise=np.random.normal(loc=0.0, scale=float(i)+10., size=self.AC_sim_CTD.shape)
            temp_sim = self.AC_sim_CTD + noise
            sim_hist, sim_bin_edges = np.histogram(temp_sim, bins=700, range=(-350.,350.), density=True)
            bin_centers = (sim_bin_edges[1:] + sim_bin_edges[:-1])/2.
            p0 = [500., -200., 30.]
            popt, pcov = curve_fit(dat2CTD.gauss, bin_centers, sim_hist, p0=p0)
            noise_mu_sigma_AC.append([(float(i)+10.), popt[1], popt[2]])
        noise_mu_sigma_AC = np.array(noise_mu_sigma_AC)

        noise_mu_sigma_DC = []
        for i in range(100):
            noise=np.random.normal(loc=0.0, scale=float(i)+10., size=self.DC_sim_CTD.shape)
            temp_sim = self.DC_sim_CTD + noise
            sim_hist, sim_bin_edges = np.histogram(temp_sim, bins=700, range=(-350.,350.), density=True)
            bin_centers = (sim_bin_edges[1:] + sim_bin_edges[:-1])/2.
            p0 = [500., 125., 30.]
            popt, pcov = curve_fit(dat2CTD.gauss, bin_centers, sim_hist, p0=p0)
            noise_mu_sigma_DC.append([(float(i)+10.), popt[1], popt[2]])
        noise_mu_sigma_DC = np.array(noise_mu_sigma_DC)

        self.slope = np.zeros((37, 37))
        self.intercept = np.zeros((37,37))

        AC_noise_match = []
        DC_noise_match = []
        for p in range(37):
            for n in range(37):
                AC_mu_obs = self.AC_params[p][n][0]
                DC_mu_obs = self.DC_params[p][n][0]
                AC_sigma_obs = self.AC_params[p][n][1]
                DC_sigma_obs = self.DC_params[p][n][1]
                AC_mu_sim = noise_mu_sigma_AC.T[1][np.argmin(np.abs(noise_mu_sigma_AC.T[2]-AC_sigma_obs))]
                DC_mu_sim = noise_mu_sigma_DC.T[1][np.argmin(np.abs(noise_mu_sigma_DC.T[2]-DC_sigma_obs))]

                if AC_mu_obs!=0 and DC_mu_obs!=0:
                    AC_noise_match.append(noise_mu_sigma_AC.T[0][np.argmin(np.abs(noise_mu_sigma_AC.T[2]-AC_sigma_obs))])
                    DC_noise_match.append(noise_mu_sigma_DC.T[0][np.argmin(np.abs(noise_mu_sigma_DC.T[2]-DC_sigma_obs))])

                self.slope[p][n] = ((AC_mu_obs - DC_mu_obs)/(AC_mu_sim - DC_mu_sim)) * (AC_mu_obs!=0.0) * (DC_mu_obs!=0.0)
                self.intercept[p][n] = (((AC_mu_obs+DC_mu_obs) - self.slope[p][n]*(AC_mu_sim + DC_mu_sim))/2.) * (AC_mu_obs!=0.0) * (DC_mu_obs!=0.0)

        ### Save a noise parameter for the AC and DC sides which corresponds to the mean over all pixels
        ### of the noise required to make the simulations match the observed Am241 CTDs.
        # print(str(round(np.mean(AC_noise_match), 1)) + ' +/- ' + str(round(np.std(AC_noise_match), 1)))
        # print(str(round(np.mean(DC_noise_match), 1)) + ' +/- ' + str(round(np.std(DC_noise_match), 1)))
        self.AC_noise = np.mean(AC_noise_match)
        self.DC_noise = np.mean(DC_noise_match)
        
        ### Some pixels were unable to be fit due to low counts. Take the mean of adjacent good pixels to guess the slope
        slope_buff = np.array([[0.0 for p in range(39)] for n in range(39)])
        slope_buff[1:-1, 1:-1] = self.slope
        for p in range(37):
            for n in range(37):
                if self.slope[p][n]==0.0:
                    buff_block = slope_buff[p:p+3][n:n+3]
                    mask = (buff_block!=0.0)
                    self.slope[p][n] = np.mean(buff_block[mask])

        ### Some pixels were unable to be fit due to low counts. Take the mean of adjacent good pixels to guess the intercept
        intercept_buff = np.array([[0.0 for p in range(39)] for n in range(39)])
        intercept_buff[1:-1, 1:-1] = self.intercept
        for p in range(37):
            for n in range(37):
                if self.intercept[p][n]==0.0:
                    buff_block = intercept_buff[p:p+3][n:n+3]
                    mask = (buff_block!=0.0)
                    self.intercept[p][n] = np.mean(buff_block[mask])

        ### Save a depth calibration file if a path is provided.
        ### Currently, depth calibration with nuclearizer doesn't work for the spare detector, so don't use this file yet.
        if savefile is not None:
            pix_code = []
            pix_stretch = []
            pix_offset = []
            for p in range(37):
                for n in range(37):
                    ### p=AC=y strips I'm pretty sure... (WRONG! n=AC=y)
                    pix_code.append(int(110000 + 100*(p+1) + n+1))
                    pix_stretch.append(self.slope[p][n])
                    pix_offset.append(self.intercept[p][n])
            pix_code = np.array(pix_code)
            pix_stretch = np.array(pix_stretch)[np.argsort(pix_code)]
            pix_offset = np.array(pix_offset)[np.argsort(pix_code)]
            pix_code = np.sort(pix_code)
            with open(savefile, 'w') as file:
                for i in range(len(pix_code)):
                    file.write(str(pix_code[i]) + '   ' + str(pix_stretch[i]) + '   ' + str(pix_offset[i]) + '   0.035   1.0\n')
        
        ### read in the simulated CTD -> depth file.     
        sim_ctd = []
        sim_depth = []
        with open(sim_file) as file:
            for line in file:
                sim_ctd.append(float(line.split(',')[1]))
                sim_depth.append(float(line.split(',')[0]))
        self.sim_ctd = np.array(sim_ctd)
        self.sim_depth = np.array(sim_depth)
        
        self.zmin = np.min(sim_depth)
        self.zmax = np.max(sim_depth)

        ### Because CTD in not monotonic, there are regions on the edges of the detector where we can't interpolate.
        ### Determine these regions and don't interpolate there.
        sim_ctd_diff = self.sim_ctd[1:] - self.sim_ctd[:-1]
        diff_midpoint = int(len(sim_ctd_diff)/2)
        # print(diff_midpoint)
        
        ### Determine the first index where monotonicity begins
        self.start = 1
        while sim_ctd_diff[:diff_midpoint][-self.start]!=0.0:
            self.start+=1
        self.start = diff_midpoint-self.start + 1
        
        ### Determine the last index to consider before monotonicity stops
        self.end = 0
        while sim_ctd_diff[diff_midpoint:][self.end]!=0.0:
            self.end+=1
        self.end = diff_midpoint+self.end
        
        # print(start)
        # print(end)
        # sim_interp = interp1d(sim_ctd[start:end+1], sim_depth[start:end+1], bounds_error=False)
        self.sim_interp = CubicSpline(np.flip(self.sim_ctd[self.start:self.end+1]), np.flip(self.sim_depth[self.start:self.end+1]), extrapolate=True)

    def depth_from_timing(self, p_strip, n_strip, p_time, n_time):
        ### Map the p and n times to a depth. Also returns a stretched and offset ctd
        ### This function is superseded by depth_from_timing_prob.
        p_time = np.array(p_time)
        n_time = np.array(n_time)

        ### map the observed ctd to the simulated curve using the stretch and offset for the specified pixel
        ctd_obs = (p_time + (5.*np.random.rand(*p_time.shape))) - (n_time + (5.*np.random.rand(*n_time.shape)))
        ### Note that strip numbers that come out of nuclearizer are 1-indexed
        ctd_stretch = (ctd_obs-self.intercept[p_strip-1][n_strip-1])/(self.slope[p_strip-1][n_strip-1])

        ### Map the simulated ctd to depth
        depth = self.sim_interp(ctd_stretch)

        ### ctds that fall outside the range of monotonicity will be assigned a depth corresponding to 
        ### the center of the region where monotonicity is not maintained.
        edge_high_mask = ctd_stretch < self.sim_ctd[self.end]
        edge_low_mask = ctd_stretch > self.sim_ctd[self.start]
        depth[edge_low_mask] = np.mean(self.sim_depth[:self.start])
        depth[edge_high_mask] = np.mean(self.sim_depth[self.end+1:])
        return ctd_obs, ctd_stretch, depth

    ### Map the p and n times to a depth. Also returns a stretched and offset ctd
    def depth_from_timing_prob(self, p_strip, n_strip, p_time, n_time, energy_p):
        ### Probabilistic depth estimate
        p_time = np.array(p_time)
        n_time = np.array(n_time)

        ### map the observed ctd to the simulated curve using the stretch and offset for the specified pixel
        ### Note that a float between 0 and 5 is added to each because of the CC readout.
        ctd_obs = (p_time + (5.*np.random.rand(*p_time.shape))) - (n_time + (5.*np.random.rand(*n_time.shape)))
        ### Strip numbers that come out of nuclearizer are 1-indexed
        ctd_stretch = (ctd_obs-self.intercept[p_strip-1][n_strip-1])/(self.slope[p_strip-1][n_strip-1])

        ### The noise on each event is energy dependent.
        ### TODO: Double check this relation. Currently just a rough estimate.
        m = (self.AC_noise - 12.)/((1./source_dict['Am241']) - (1./source_dict['Cs137']))
        b = self.AC_noise - (m/source_dict['Am241'])
        noise = b + m/(energy_p.values)

        bad = np.logical_or(ctd_stretch > (np.max(self.sim_ctd) + 2*noise),ctd_stretch < (np.min(self.sim_ctd) - 2*noise))
        
        ### Map the simulated ctd to depth
        depth = []
        depth_err = []
        for i, tau in enumerate(ctd_stretch):
            ### Calculate the probability distribution with depth given measured tau and noise
            prob_dist = stats.norm.pdf(self.sim_ctd, loc=tau, scale=noise[i])
            mean_depth = np.sum(prob_dist*self.sim_depth)/np.sum(prob_dist)
            sigma_depth = np.sqrt(np.sum(prob_dist*np.square(self.sim_depth-mean_depth))/np.sum(prob_dist))
            
            prob_norm = (np.sum(prob_dist) - ((prob_dist[0] + prob_dist[-1])/2.))

            ### Assign the depth randomly given the probability distribution
            prob_dist = prob_dist/prob_norm
            cdf = np.concatenate([[0.0], (np.cumsum(prob_dist)-((prob_dist + prob_dist[0])/2.0))[1:]])
            depth.append(interp1d(cdf, self.sim_depth)(np.random.uniform()))
            
            ### Assign 1-sigma depth error given the probability distribution
            depth_err.append(sigma_depth)

        return ctd_obs, ctd_stretch, np.array(depth), np.array(depth_err), bad

    ### Map the p and n times to a depth. Also returns a stretched and offset ctd
    def depth_from_timing_and_energy(self, p_strip, n_strip, p_time, n_time, energy_p, energy_n, sim_dCCE_path, ae_over_ah, b, c):
        ### Probabilistic depth estimate
        p_time = np.array(p_time)
        n_time = np.array(n_time)

        sim_dCCE = np.genfromtxt(sim_dCCE_path, delimiter=',')

        ### map the observed ctd to the simulated curve using the stretch and offset for the specified pixel
        ### Note that a float between 0 and 5 is added to each because of the CC readout.
        ctd_obs = (p_time + (5.*np.random.rand(*p_time.shape))) - (n_time + (5.*np.random.rand(*n_time.shape)))
        ### Strip numbers that come out of nuclearizer are 1-indexed
        ctd_stretch = (ctd_obs-self.intercept[p_strip-1][n_strip-1])/(self.slope[p_strip-1][n_strip-1])

        ### The noise on each event is energy dependent.
        ### TODO: Double check this relation. Currently just a rough estimate.
        m = (self.AC_noise - 12.)/((1./source_dict['Am241']) - (1./source_dict['Cs137']))
        d = self.AC_noise - (m/source_dict['Am241'])
        noise = d + m/(energy_p.values)

        bad = np.logical_or(ctd_stretch > (np.max(self.sim_ctd) + 2*noise),ctd_stretch < (np.min(self.sim_ctd) - 2*noise))

        e_cce = (1.-b*sim_dCCE[1][::-1])*(1.-c*sim_dCCE[2][::-1])
        h_cce = (1.-b*sim_dCCE[3][::-1])*(1.-c*sim_dCCE[4][::-1])
        depth_to_eratio = UnivariateSpline(sim_dCCE[0], ae_over_ah*(e_cce/h_cce))
        eratio = energy_p.values/energy_n.values
        
        ### Map the simulated ctd to depth
        depth = []
        depth_err = []
        for i, tau in enumerate(ctd_stretch):
            ### Calculate the probability distribution with depth given measured tau and noise
            prob_dist = stats.norm.pdf(self.sim_ctd, loc=tau, scale=noise[i]) * stats.norm.pdf(depth_to_eratio(self.sim_depth), loc=eratio[i], scale=0.004)
            mean_depth = np.sum(prob_dist*self.sim_depth)/np.sum(prob_dist)
            sigma_depth = np.sqrt(np.sum(prob_dist*np.square(self.sim_depth-mean_depth))/np.sum(prob_dist))
            
            prob_norm = (np.sum(prob_dist) - ((prob_dist[0] + prob_dist[-1])/2.))

            ### Assign the depth randomly given the probability distribution
            prob_dist = prob_dist/prob_norm
            cdf = np.concatenate([[0.0], (np.cumsum(prob_dist)-((prob_dist + prob_dist[0])/2.0))[1:]])
            depth.append(interp1d(cdf, self.sim_depth)(np.random.uniform()))
            
            ### Assign 1-sigma depth error given the probability distribution
            depth_err.append(sigma_depth)

        return ctd_obs, ctd_stretch, depth, depth_err, bad



    def get_simdata(self):
        return self.sim_depth, self.sim_ctd

class Detector:
### Work in progress
    def __init__(self, irradiation_list, zmin, zmax, calibrator=None):
        self.irradiation_list = irradiation_list
        self.calibrator = calibrator
        self.ae = None
        self.ah = None
        self.b = None
        self.c = None
        self.zmin = zmin
        self.zmax = zmax

class Irradiation:
### Work in progress
    def __init__(self, source, irr_side, filelist, emin = 30., emax=10000.):
        
        self.source = None
        if s in source_dict:
            if s[:2] == source.strip()[:2]:
                self.source = s
        if self.source is None:
            raise Exception('The provided source was not recognized.')

        if irr_side.strip().upper() in ['AC', 'DC', 'OTHER']:
            self.irr_side = irr_side.strip().upper()
        else:
            raise Exception('The irradiation side was not recognized. Valid options are \"AC\", \"DC\", or \"other\"')

        self.emin = emin
        self.emax = emax        

        self.data = None
        self.load_data(filelist, emin=self.emin, emax=self.emax)
        self.cce_corrected = False
        self.spline_corrected = False
        self.depth_calibrated = False
        self.ctd_obs = None
        self.ctd_stretch = None

    def load_data(self, filelist, emin, emax):
        # SH det  side  strip_num (starting at 1)  did strip trigger (0 or 1)  
        # timing raw AD counter units  corrected AD counter units  ADC  energy  ??  
        # HT x y z energy [cm, cm, cm, keV]

        rows = []
        col_names = ["det","strip_p","energy_p", "time_p", "strip_n","energy_n", "time_n", "x","y","z", "z_err", "bad"]
        for file in filelist:
            with open(file, "r") as f:
                
                ev_block = []
                
                for line in f:
                #for each line, start a block of lines corresponding to an event
                    
                    if line.startswith('SE'):
                    # If the accumulated block has 6 lines then it's a single-pixel event. 
                    # Allowing for "bad pairing" events to recover events with extreme energy differences due to trapping.
                        if (len(ev_block) == 6 and np.product(np.array([("BD" not in l) for l in ev_block]))) or \
                        (len(ev_block) == 7 and np.sum(np.array([("bad pairing" in l) for l in ev_block]))):
                                

                                if len(ev_block[3].split(" ")) < 9:
                                       print(ev_block)
                                if len(ev_block[4].split(" ")) < 9:
                                       print(ev_block)

                                energy_p = float(ev_block[3].split(" ")[8])
                                energy_n = float(ev_block[4].split(" ")[8])

                                time_p = float(ev_block[3].split(" ")[5])
                                time_n = float(ev_block[4].split(" ")[5])

                                # select photopeak events
                                ### Allow DC energy to be lower than the minimum to account for trapping.
                                # if (energy_p < e_max and energy_p > e_min) and (energy_n < e_max and energy_n > e_min) and (np.abs(time_p)<400) and (np.abs(time_n)<400):
                                if (energy_p < e_max and energy_p > e_min) and (energy_n < e_max) and (np.abs(time_p)<400) and (np.abs(time_n)<400):

                                    # save info from SH p line
                                    det = int(ev_block[3].split(" ")[1])
                                    strip_p = int(ev_block[3].split(" ")[3])

                                    # save info from SH n line
                                    strip_n = int(ev_block[4].split(" ")[3])

                                    # save position [cm] info from HT line
                                    x = float(ev_block[5].split(" ")[1])
                                    y = float(ev_block[5].split(" ")[2])
                                    z = float(ev_block[5].split(" ")[3])

                                    # save info to df
                                    columns = [det,strip_p,energy_p,time_p,strip_n,energy_n,time_n,x,y,z, 0.0, False]
                                    rows.append(columns)
                        ev_block = []
                        
                    else:
                        ev_block.append(line)
        
        df = pd.DataFrame(rows,columns=col_names)
        if self.data is None:
            self.data = df
        else:
            self.data = pd.concat(self.data, df)
        return df

    def calibrate_depth(self, calibrator, n_strips=37):
        self.ctd_obs = [[[] for p in range(n_strips)] for n in range(n_strips)]
        self.ctd_stretch = [[[] for p in range(n_strips)] for n in range(n_strips)]

        for p in range(n_strips):
            for n in range(n_strips):
                ctd_obs_temp, ctd_stretch_temp, depth, depth_err, bad = calibrator.depth_from_timing_prob(p+1, n+1, self.data.loc[self.data.strip_p.eq(p+1)&self.data.strip_n.eq(n+1), 'time_p'], \
                                                                                           self.data.loc[self.data.strip_p.eq(p+1)&self.data.strip_n.eq(n+1), 'time_n'], \
                                                                                          self.data.loc[self.data.strip_p.eq(p+1)&self.data.strip_n.eq(n+1), 'energy_p'])
                self.data.loc[self.data.strip_p.eq(p+1)&self.data.strip_n.eq(n+1), 'z'] =  depth
                self.data.loc[self.data.strip_p.eq(p+1)&self.data.strip_n.eq(n+1), 'z_err'] =  depth_err
                self.ctd_obs[p][n] = ctd_obs_temp[~bad]
                self.ctd_stretch[p][n] = ctd_stretch_temp[~bad]
        return


def make_df_from_dat(files, e_min = 550., e_max = 700.):
    
    # SH det  side  strip_num (starting at 1)  did strip trigger (0 or 1)  
           #timing raw AD counter units  corrected AD counter units  ADC  energy  ??  
    # HT x y z energy [cm, cm, cm, keV]   

    rows = []
    col_names = ["ID","det","strip_p","energy_p", "time_p", "strip_n","energy_n", "time_n", "x","y","z", "z_err", "bad"]
    for file in files:
        with open(file, "r") as f:
            
            ev_block = []
            
            for line in f:
            #for each line, start a block of lines corresponding to an event
                
                if line.startswith('SE'):
                # If the accumulated block has 6 lines then it's a single-pixel event. 
                # Allowing for "bad pairing" events to recover events with extreme energy differences due to trapping.
                    if (len(ev_block) == 6 and np.prod(np.array([("BD" not in l) for l in ev_block]))) or \
                    (len(ev_block) == 7 and np.sum(np.array([("bad pairing" in l) for l in ev_block]))):
                            
                            ID = int(ev_block[0].split(" ")[-1].strip("\n"))

                            if len(ev_block[3].split(" ")) < 9:
                                   print(ev_block)
                            if len(ev_block[4].split(" ")) < 9:
                                   print(ev_block)

                            energy_p = float(ev_block[3].split(" ")[8])
                            energy_n = float(ev_block[4].split(" ")[8])

                            time_p = float(ev_block[3].split(" ")[5])
                            time_n = float(ev_block[4].split(" ")[5])

                            # select photopeak events
                            ### Allow DC energy to be lower than the minimum to account for trapping.
                            # if (energy_p < e_max and energy_p > e_min) and (energy_n < e_max and energy_n > e_min) and (np.abs(time_p)<400) and (np.abs(time_n)<400):
                            if (energy_p < e_max and energy_n < e_max) and (energy_n > e_min or energy_p > e_min) and (np.abs(time_p)<400) and (np.abs(time_n)<400):

                                # save info from SH p line
                                det = int(ev_block[3].split(" ")[1])
                                strip_p = int(ev_block[3].split(" ")[3])

                                # save info from SH n line
                                strip_n = int(ev_block[4].split(" ")[3])

                                # save position [cm] info from HT line
                                x = float(ev_block[5].split(" ")[1])
                                y = float(ev_block[5].split(" ")[2])
                                z = float(ev_block[5].split(" ")[3])

                                # save info to df
                                columns = [ID,det,strip_p,energy_p,time_p,strip_n,energy_n,time_n,x,y,z,0.0, False]
                                rows.append(columns)
                    ev_block = []
                    
                else:
                    ev_block.append(line)
    
    df = pd.DataFrame(rows,columns=col_names)
    return df

def make_depthplot(df, plot_suffix, z_bins, plot_dir="/Users/Sophie/Desktop/COSI/calibrations/Data/trapping_figures/", source='Cs137'):
    ### Take in a depth-calibrated dataset, bin events in depth, and fit the spectra.
    ### Produce and return depth plots.
    if source not in source_dict:
        print('Source not recognized. Must be one of the following: Am241, Cs137, Ba133, Co57')
    line_e = source_dict[source]

    num_z = len(z_bins)-1
    df["z_binned"] = pd.cut(df["z"],bins=z_bins)

    centroid_list = []
    centroid_err_list = []

    tail_cnts = []
    gauss_cnts = []
    tail_frac = []

    emin_list = []
    emax_list = []

    fig, axes = plt.subplots(figsize = (12, num_z*2.), nrows=num_z*2, ncols=2, sharex=True, sharey='row', \
                            gridspec_kw={'hspace':0, 'wspace':0, 'height_ratios':np.array([(2,1) for i in range(num_z)]).flatten()})
    for i in range(num_z):
        
        label = "{}-{} cm".format(round(z_bins[i], 2),round(z_bins[i+1], 2))

        # fit both sides
        temp_x0 = []
        temp_x0_err = []
        for j, side in enumerate(['p', 'n']):
            if side=='p':
                carrier='HV'
            else:
                carrier='LV'

            ax = axes[i*2][j]
            resid_ax = axes[(i*2)+1][j]            
            color = 'C' + str(j)
            
            energies = df.loc[df.z.le(z_bins[i+1])&df.z.gt(z_bins[i]), 'energy_'+side].values
            temp_emin = stats.mode(np.floor(energies))[0] - 30.
            temp_emax = stats.mode(np.floor(energies))[0] + 20.
            emin_list.append(temp_emin)
            emax_list.append(temp_emax)
            emask = (energies < temp_emax) * (energies > temp_emin)
            energies = energies[emask]
            hist,binedges,_ = ax.hist(energies, histtype="step",color=color,bins=100,label=carrier + " signal, " + label)
            bin_centers = np.array((binedges[:-1] + binedges[1:]) / 2)

            c = cost.UnbinnedNLL(energies, gauss_plus_tail_pdf)

            m = Minuit(c, BoverA=0.5, x0=bin_centers[np.argmax(hist)], sigma_gauss=1.2, gamma=global_gamma, CoverB=global_CoverB, D=global_D, sigma_ratio=global_sigma_ratio, Emin=temp_emin, Emax=temp_emax)
            m.limits["x0"] = (bin_centers[np.argmax(hist)]-3., bin_centers[np.argmax(hist)]+3.)
            m.limits["BoverA", "sigma_gauss"] = (0, None)
            m.fixed["gamma", "CoverB", "D", "sigma_ratio", "Emin", "Emax"] = True
            # m.errors["x0", "BoverA", "sigma_gauss"] = np.array(m.values["x0", "BoverA", "sigma_gauss"]) * 0.1
            # m.scan()
            m.migrad()
            m.hesse()
            # m.minos("x0")

            BoverA, x0, sigma_gauss = m.values[:3]
            A = np.sum(hist)*(bin_centers[1]-bin_centers[0])/\
                quad(gauss_plus_tail, np.min(energies), np.max(energies), args = (BoverA, x0, sigma_gauss, global_gamma, global_CoverB, global_D, global_sigma_ratio))[0]
            # print(m.values)
            # print(m.errors['x0'])
            temp_x0.append(m.values['x0'])
            ### Record error either from fitting or using an estimate of gaussian centroid error, whichever is bigger.
            temp_x0_err.append(max(m.errors['x0'], np.sqrt(m.values["sigma_gauss"]/(A*np.sqrt(2.*np.pi)))))
            # temp_x0_err.append(m.errors['x0'])
            # print(c(*m.values))
            # print(np.sum(np.log(gauss_plus_tail_pdf(df_dat_det[df_dat_det["z_binned"]==z]["energy_p"].values, *m.values))))
            B = A*BoverA
            C = B*global_CoverB

            ax.plot(bin_centers,A*gauss_plus_tail(bin_centers, BoverA, x0, sigma_gauss, global_gamma, global_CoverB, global_D, global_sigma_ratio),color= color, lw=0.5)
            resid_ax.errorbar(bin_centers[hist>0], \
                               (hist[hist>0] - A*gauss_plus_tail(bin_centers[hist>0],BoverA, x0, sigma_gauss, global_gamma, global_CoverB, global_D, global_sigma_ratio))/np.sqrt(hist[hist>0]), \
                               xerr=(bin_centers[1]-bin_centers[0])/2, yerr=1., ls='', lw=0.6, color=color)
            ax.plot(bin_centers,A*gaussian(bin_centers, x0, sigma_gauss),color= color, ls='--', lw=0.5)
            ax.plot(bin_centers, B*exp_tail(bin_centers, x0, gamma=global_gamma)*shelf(bin_centers, x0, sigma_gauss*global_sigma_ratio),color= color, ls='--', lw=0.5)
            ax.plot(bin_centers, C*linear_tail(bin_centers, x0, global_D)*shelf(bin_centers, x0, sigma_gauss*global_sigma_ratio),color= color, ls='--', lw=0.5)
            ax.axvline(x0, ls='--', color='C3')
            ax.axvline(line_e, ls='--', color='C2')
            if j==0:
                ax.set_ylabel("Counts")
            ax.set_yscale('log')
            ax.set_ylim((np.min(hist) + 10., 2.0*np.max(hist)))
            ax.legend(loc=2, fontsize=12)
            resid_ax.axhline(0, color='red', lw=0.5)

        axes[(i*2)+1][0].set_ylabel(r'$\chi$')
        centroid_list.append(temp_x0)
        centroid_err_list.append(temp_x0_err)

    axes[-1][0].set_xlim(np.min(emin_list), np.max(emax_list))
    axes[-1][1].set_xlim(np.min(emin_list), np.max(emax_list))
    axes[-1][0].set_xlabel("Energy (keV)")
    axes[-1][1].set_xlabel("Energy (keV)")
    plt.tight_layout()

    plt.savefig(plot_dir + 'zplitspectra_uncorr_' + plot_suffix + '.pdf')
    plt.close()

    centroid_list = np.array(centroid_list)/line_e
    centroid_err_list = np.array(centroid_err_list)/line_e

    return z_bins, [centroid_list.T[0], centroid_err_list.T[0]], [centroid_list.T[1], centroid_err_list.T[1]]



def depth_correction(df, z_bins, e_trapping, h_trapping, plot_dir="/Users/Sophie/Desktop/COSI/calibrations/Data/trapping_figures/", plot_suffix='', source='Cs137'):
    ### Correct measured energy on an event-by-event basis using the depth plots produced by make_depthplot.
    ### Return the dataset with corrected energies in new columns.
    if source not in source_dict:
        print('Source not recognized. Must be one of the following: Am241, Cs137, Ba133, Co57')
    line_e = source_dict[source]

    z_list = (z_bins[:-1] + z_bins[1:])/2.
    z_err = (z_bins[1:]-z_bins[:-1])/2.
    splines = {'p': UnivariateSpline(z_list, e_trapping[0]), 'n': UnivariateSpline(z_list, h_trapping[0])}
    plt.figure()
    xs = np.linspace(z_bins[0], z_bins[-1])
    plt.plot(xs, splines['p'](xs)*line_e, zorder=0, color='C0', ls = '--', lw = 0.75)
    plt.plot(xs, splines['n'](xs)*line_e, zorder=0, color='C1', ls = '--', lw = 0.75)
    plt.errorbar(z_list, e_trapping[0]*line_e, xerr = z_err, yerr=e_trapping[1]*line_e, fmt=".", label="HV signal", color='C0')
    plt.errorbar(z_list, h_trapping[0]*line_e, xerr = z_err, yerr=h_trapping[1]*line_e, fmt=".", label="LV signal", color='C1')
    plt.axhline(line_e, ls='--', color='C2')
    # plt.legend(loc=4)
    plt.xlabel("Detector Depth (cm)"); plt.ylabel("Centroid Energy (keV)")
    plt.tight_layout()
    plt.savefig(plot_dir + 'e_hole_trapping_' + plot_suffix + '.pdf')
    plt.close()

    fig, axes = plt.subplots(figsize = (12, 9), nrows=2, ncols=2, sharex=True, sharey=False, gridspec_kw={'hspace':0, 'wspace':0})

    for i, side in enumerate(splines):
        if side=='p':
            carrier='HV'
        else:
            carrier='LV'
        
        color = 'C'+str(i)
        
        ax = axes[0][i]
        energies = df['energy_'+side].values

        hist,binedges,_ = ax.hist(energies[~df['bad'].values], histtype="step", bins=100, range=(line_e-25., line_e+10.), color=color, label="Uncorrected " + carrier + " signal")
        bin_centers = np.array((binedges[:-1] + binedges[1:]) / 2)

        fwhm, fwtm, fwhm_err, fwtm_err = get_FWHM_FWTM(bin_centers, hist)
        print('FWHM = ' + str(round(fwhm, 3)) + '+/-' + str(round(fwhm_err, 3)))
        print('FWTM = ' + str(round(fwtm, 3)) + '+/-' + str(round(fwtm_err, 3)))

        ax.axvline(line_e, ls='--', color='red')
        if i==0:
            ax.set_ylabel("Counts")
        ax.set_yscale('log')
        ax.set_ylim(bottom = np.max(hist)/100., top = 3.9*np.max(hist))
        ax.legend(loc=2)
        # ax.text(0.1, 0.8, 'FWHM = ' + str(round(fwhm, 1)) + r'$\pm$' + str(round(fwhm_err, 1)) + ' keV', transform = ax.transAxes)
        # ax.text(0.1, 0.75, 'FWTM = ' + str(round(fwtm, 1)) + r'$\pm$' + str(round(fwhm_err, 1)) + ' keV', transform = ax.transAxes)
        ax.text(0.1, 0.8, 'FWHM = ' + str(round(fwhm, 1)), transform = ax.transAxes)
        ax.text(0.1, 0.75, 'FWTM = ' + str(round(fwtm, 1)), transform = ax.transAxes)


        ### Correct the measured energies according to the CCE spline
        energies = df["energy_"+side].values/splines[side](df['z'].values)
        df['depth_corrected_energy_'+side] = energies

        ax = axes[1][i]
        hist,binedges,_ = ax.hist(energies[~df['bad'].values], histtype="step", bins=100, range=(line_e-25., line_e+10.), color=color, label="Corrected " + carrier + " signal")
        bin_centers = np.array((binedges[:-1] + binedges[1:]) / 2)

        fwhm, fwtm, fwhm_err, fwtm_err = get_FWHM_FWTM(bin_centers, hist)
        print('FWHM = ' + str(round(fwhm, 3)) + '+/-' + str(round(fwhm_err, 3)))
        print('FWTM = ' + str(round(fwtm, 3)) + '+/-' + str(round(fwtm_err, 3)))

        ax.axvline(line_e, ls='--', color='C2')
        if i==0:
            ax.set_ylabel("Counts")
        ax.set_yscale('log')
        ax.set_ylim(bottom = np.max(hist)/100., top = 3.9*np.max(hist))
        ax.legend(loc=2)
        # ax.text(0.1, 0.8, 'FWHM = ' + str(round(fwhm, 1)) + r'$\pm$' + str(round(fwhm_err, 1)) + ' keV', transform = ax.transAxes)
        # ax.text(0.1, 0.75, 'FWTM = ' + str(round(fwtm, 1)) + r'$\pm$' + str(round(fwhm_err, 1)) + ' keV', transform = ax.transAxes)
        ax.text(0.1, 0.8, 'FWHM = ' + str(round(fwhm, 1)), transform = ax.transAxes)
        ax.text(0.1, 0.75, 'FWTM = ' + str(round(fwtm, 1)), transform = ax.transAxes)



    axes[1][0].set_xlabel("Energy (keV)")
    axes[1][1].set_xlabel("Energy (keV)")
    plt.tight_layout()

    plt.savefig(plot_dir + 'depth_corrected_spectra_' + plot_suffix + '.pdf')
    plt.close()

    return df

def depth_correction_CCE(df, ae, ah, b, c, sim_dCCE_path, plot_dir="//Users/Sophie/Desktop/COSI/calibrations/Data/trapping_figures/", plot_suffix='', source='Cs137'):
    ### Correct measured energy on an event-by-event basis using the charge collection efficiency parameters.
    ### Return the dataset with corrected energies in new columns.
    if source not in source_dict:
        print('Source not recognized. Must be one of the following: Am241, Cs137, Ba133, Co57')
    line_e = source_dict[source]

    sim_dCCE = np.genfromtxt(sim_dCCE_path, delimiter=',')

    ### TODO: double check why I flipped the y values. Presumably p and n are flipped compared to simulations.
    e_cce = UnivariateSpline(sim_dCCE[0], ae*(1.-b*sim_dCCE[1][::-1])*(1.-c*sim_dCCE[2][::-1]))
    h_cce = UnivariateSpline(sim_dCCE[0], ah*(1.-b*sim_dCCE[3][::-1])*(1.-c*sim_dCCE[4][::-1]))

    cces = {'p': e_cce, 'n': h_cce}

    fig, axes = plt.subplots(figsize = (12, 9), nrows=2, ncols=2, sharex=True, sharey=False, gridspec_kw={'hspace':0, 'wspace':0})

    for side in cces:

        i=0
        if side=='p':
            carrier='HV'
        else:
            carrier='LV'
            i=1

        color = 'C'+str(i)
        
        ax = axes[0][i]
        energies = df['energy_'+side].values

        hist,binedges,_ = ax.hist(energies[~df['bad'].values], histtype="step", bins=100, label="Uncorrected " + carrier + " signal", range=(erange_dict[source][0], erange_dict[source][1]), color=color)
        bin_centers = np.array((binedges[:-1] + binedges[1:]) / 2)

        # fwhm_spline = UnivariateSpline(bin_centers, hist-0.5*np.max(hist))
        # fwtm_spline = UnivariateSpline(bin_centers, hist-0.1*np.max(hist))
        # fwhm = fwhm_spline.roots()[-1]-fwhm_spline.roots()[0]
        # fwtm = fwtm_spline.roots()[-1]-fwtm_spline.roots()[0]

        fwhm, fwtm, fwhm_err, fwtm_err = get_FWHM_FWTM(bin_centers, hist)
        print('FWHM = ' + str(round(fwhm, 3)) + '+/-' + str(round(fwhm_err, 3)))
        print('FWTM = ' + str(round(fwtm, 3)) + '+/-' + str(round(fwtm_err, 3)))

        ax.axvline(line_e, ls='--', color='red')
        if i==0:
            ax.set_ylabel("Counts")
        ax.set_yscale('log')
        ax.set_ylim(bottom = np.max(hist)/100., top = 3.9*np.max(hist))
        ax.legend(loc=2)
        # ax.text(0.05, 0.78, 'FWHM = ' + str(round(fwhm, 1)) + r'$\pm$' + str(round(fwhm_err, 1)) + ' keV', transform = ax.transAxes)
        # ax.text(0.05, 0.72, 'FWTM = ' + str(round(fwtm, 1)) + r'$\pm$' + str(round(fwhm_err, 1)) + ' keV', transform = ax.transAxes)
        ax.text(0.05, 0.78, 'FWHM = ' + str(round(fwhm, 1)), transform = ax.transAxes)
        ax.text(0.05, 0.72, 'FWTM = ' + str(round(fwtm, 1)), transform = ax.transAxes)


        ### Correct the measured energies according to the CCE spline
        energies = df["energy_"+side].values/cces[side](df['z'].values)
        df['depth_corrected_energy_'+side] = energies

        ax = axes[1][i]
        hist,binedges,_ = ax.hist(energies[~df['bad'].values], histtype="step", bins=100, label="Corrected " + carrier + " signal", range=(erange_dict[source][0], erange_dict[source][1]), color=color)
        bin_centers = np.array((binedges[:-1] + binedges[1:]) / 2)

        fwhm, fwtm, fwhm_err, fwtm_err = get_FWHM_FWTM(bin_centers, hist)
        print('FWHM = ' + str(round(fwhm, 3)) + '+/-' + str(round(fwhm_err, 3)))
        print('FWTM = ' + str(round(fwtm, 3)) + '+/-' + str(round(fwtm_err, 3)))

        ax.axvline(line_e, ls='--', color='red')
        if i==0:
            ax.set_ylabel("Counts")
        ax.set_yscale('log')
        ax.set_ylim(bottom = np.max(hist)/100., top = 3.9*np.max(hist))
        ax.legend(loc=2)
        # ax.text(0.05, 0.78, 'FWHM = ' + str(round(fwhm, 1)) + r'$\pm$' + str(round(fwhm_err, 1)) + ' keV', transform = ax.transAxes)
        # ax.text(0.05, 0.72, 'FWTM = ' + str(round(fwtm, 1)) + r'$\pm$' + str(round(fwhm_err, 1)) + ' keV', transform = ax.transAxes)
        ax.text(0.05, 0.78, 'FWHM = ' + str(round(fwhm, 1)), transform = ax.transAxes)
        ax.text(0.05, 0.72, 'FWTM = ' + str(round(fwtm, 1)), transform = ax.transAxes)


    axes[1][0].set_xlabel("Energy (keV)")
    axes[1][1].set_xlabel("Energy (keV)")
    plt.tight_layout()

    plt.savefig(plot_dir + 'CCEdepth_corrected_spectra_' + plot_suffix + '.pdf')
    plt.close()

    return df




def fit_CCE(z_bins, e_trapping, h_trapping, sim_dCCE_path, plot_dir="/Users/Sophie/Desktop/COSI/calibrations/Data/trapping_figures/", plot_suffix='', source='Cs137', trim_index=0):
    ### Fit the measured depth plots to simulated CCE curves in order to estimate trapping lengths.

    if source not in source_dict:
        print('Source not recognized. Must be one of the following: Am241, Cs137, Ba133, Co57')
    line_e = source_dict[source]

    sim_dCCE = np.genfromtxt(sim_dCCE_path, delimiter=',')
    # print(sim_dCCE)

    def e_depth_plot(z, ae, b, c):
        CCE = ae*(1.-b*sim_dCCE[1][::-1])*(1.-c*sim_dCCE[2][::-1])
        return UnivariateSpline(sim_dCCE[0], CCE)(z)

    def h_depth_plot(z, ah, b, c):
        CCE = ah*(1.-b*sim_dCCE[3][::-1])*(1.-c*sim_dCCE[4][::-1])
        return UnivariateSpline(sim_dCCE[0], CCE)(z)

    z_list = (z_bins[:-1] + z_bins[1:])/2.
    z_err = (z_bins[1:]-z_bins[:-1])/2.

    c = cost.LeastSquares(z_list[trim_index:-trim_index], e_trapping[0][trim_index:-trim_index], e_trapping[1][trim_index:-trim_index], e_depth_plot) + \
    cost.LeastSquares(z_list[trim_index:-trim_index], h_trapping[0][trim_index:-trim_index], h_trapping[1][trim_index:-trim_index], h_depth_plot)

    m = Minuit(c, ae=np.max(e_trapping[0]), ah=np.max(h_trapping[0]), b=1.0, c=9.)
    m.limits["b", "c"] = (0, None)
    # m.errors["b", "c"] = 0.1
    # m.scan()
    m.migrad()
    m.hesse()
    m.minos()

    plt.figure()

    plt.errorbar(z_list[trim_index:-trim_index], e_trapping[0][trim_index:-trim_index]*line_e, xerr = z_err[trim_index:-trim_index], yerr=e_trapping[1][trim_index:-trim_index]*line_e, fmt=".", label="HV signal")
    plt.errorbar(z_list[trim_index:-trim_index], h_trapping[0][trim_index:-trim_index]*line_e, xerr = z_err[trim_index:-trim_index], yerr=h_trapping[1][trim_index:-trim_index]*line_e, fmt=".", label="LV signal")
    plt.errorbar(z_list[:trim_index], e_trapping[0][:trim_index]*line_e, xerr = z_err[:trim_index], yerr=e_trapping[1][:trim_index]*line_e, fmt="x", color='C0')
    plt.errorbar(z_list[:trim_index], h_trapping[0][:trim_index]*line_e, xerr = z_err[:trim_index], yerr=h_trapping[1][:trim_index]*line_e, fmt="x", color='C1')
    plt.errorbar(z_list[-trim_index:], e_trapping[0][-trim_index:]*line_e, xerr = z_err[-trim_index:], yerr=e_trapping[1][-trim_index:]*line_e, fmt="x", color='C0')
    plt.errorbar(z_list[-trim_index:], h_trapping[0][-trim_index:]*line_e, xerr = z_err[-trim_index:], yerr=h_trapping[1][-trim_index:]*line_e, fmt="x", color='C1')


    plt.plot(sim_dCCE[0], e_depth_plot(sim_dCCE[0], *m.values['ae', 'b', 'c'])*line_e, color='C0', zorder=0, lw=0.9)
    plt.plot(sim_dCCE[0], h_depth_plot(sim_dCCE[0], *m.values['ah', 'b', 'c'])*line_e, color='C1', zorder=0, lw=0.9)

    plt.axhline(line_e, ls='--', color='C2', zorder=0)
    plt.legend()
    plt.xlabel("Detector Depth (cm)"); plt.ylabel("Centroid Energy (keV)")
    plt.tight_layout()
    plt.savefig(plot_dir + 'CCE_fit_energy_' + plot_suffix + '.pdf')
    plt.close()

    plt.figure()

    plt.errorbar(z_list[trim_index:-trim_index], e_trapping[0][trim_index:-trim_index]/m.values['ae'], xerr = z_err[trim_index:-trim_index], yerr=e_trapping[1][trim_index:-trim_index]/m.values['ae'], fmt=".", label="HV signal")
    plt.errorbar(z_list[trim_index:-trim_index], h_trapping[0][trim_index:-trim_index]/m.values['ah'], xerr = z_err[trim_index:-trim_index], yerr=h_trapping[1][trim_index:-trim_index]/m.values['ah'], fmt=".", label="LV signal")
    plt.errorbar(z_list[:trim_index], e_trapping[0][:trim_index]/m.values['ae'], xerr = z_err[:trim_index], yerr=e_trapping[1][:trim_index]/m.values['ae'], fmt="x", color='C0')
    plt.errorbar(z_list[:trim_index], h_trapping[0][:trim_index]/m.values['ah'], xerr = z_err[:trim_index], yerr=h_trapping[1][:trim_index]/m.values['ah'], fmt="x", color='C1')
    plt.errorbar(z_list[-trim_index:], e_trapping[0][-trim_index:]/m.values['ae'], xerr = z_err[-trim_index:], yerr=e_trapping[1][-trim_index:]/m.values['ae'], fmt="x", color='C0')
    plt.errorbar(z_list[-trim_index:], h_trapping[0][-trim_index:]/m.values['ah'], xerr = z_err[-trim_index:], yerr=h_trapping[1][-trim_index:]/m.values['ah'], fmt="x", color='C1')

    plt.plot(sim_dCCE[0], e_depth_plot(sim_dCCE[0], *m.values['ae', 'b', 'c'])/m.values['ae'], color='C0', zorder=0, lw=0.9)
    plt.plot(sim_dCCE[0], h_depth_plot(sim_dCCE[0], *m.values['ah', 'b', 'c'])/m.values['ah'], color='C1', zorder=0, lw=0.9)

    plt.legend()
    plt.xlabel("Detector Depth (cm)")
    plt.ylabel("CCE")
    plt.ylim(top=1.0)
    plt.tight_layout()
    plt.savefig(plot_dir + 'CCE_fit_norm_' + plot_suffix + '.pdf')
    plt.close()

    return m
