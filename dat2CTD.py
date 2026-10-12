#!/usr/bin/env python

### This script takes in a calibrated .dat file and produces a CTD for single-pixel events.
### Written by Sean Pike in October 2023

import numpy as np
import matplotlib.pyplot as plt
import sys, getopt
import os
from tqdm import tqdm
import argparse
from scipy.optimize import curve_fit
from scipy.stats import norm

def gauss(x, A, mu, sigma):
    return A*norm.pdf(x, loc=mu, scale=sigma)

def dat2CTD(filelist, emin, emax, nstrips):
    ### Take in a list of calibrated .dat files and record the CTD between the p-side and n-side for events in energy range emin < E < emax.
    CTD = [[[] for i in range(nstrips)]for i in range(nstrips)]
    SH_list = []
    BD_list = []
    print('reading input files...')
    for filename in filelist:
        with open(filename) as file:
            for line in file:
                if line[0]=='S' and line[1]=='E':
                    ### If we reached a new event block, indicated by 'SE' tag, append the previous block in CTD
                    if len(BD_list)==0 and len(SH_list)==2:
                        ### If the event does not have any bad flags and only has 2 energy readings
                        if SH_list[0][2]=='l' and SH_list[1][2]=='h':
                            ##print('energy read')
                            ### If the 2 energy readings correspond to opposite sides of the detector
                            if (emin <= float(SH_list[0][8]) <= emax) and (emin <= float(SH_list[1][8]) <= emax):
                                #print('p and n present')
                                ### If the energy read on both sides of the detector lies in the specified range
                                if len(SH_list[0][5]) <=3 and len(SH_list[1][5]) <=3:
                                    print('energy in range')
                                    ### If the rise time makes sense (some are very large numbers, indicating overflow or something)
                                    ### Then record the difference between the p-side time and the n-side time
                                    p = int(SH_list[0][3])-1
                                    n = int(SH_list[1][3])-1
                                    ### Because the recorded times are in bins of 5ns, we add a random float between 0 and 5 to smooth out the bins.
                                    print('record time difference')
                                    CTD[p][n].append((float(SH_list[0][5]) + (5.*np.random.rand())) - (float(SH_list[1][5]) + (5.*np.random.rand())))
                    SH_list = []
                    BD_list = []
                else:
                    if line[0]=='S' and line[1]=='H':
                        SH_list.append(line.split())
                    if line[0]=='B' and line[1]=='D':
                        BD_list.append(line.split()[1])
    return CTD

def writeCTD(CTD, outdir, file_prefix, nstrips):
    print('writing data files...')
    for p in tqdm(range(nstrips)):
        for n in range(nstrips):
            hist, bin_edges = np.histogram(CTD[p][n], bins=np.linspace(-350,350,num=100))
            bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
            np.array([bin_centers, hist]).tofile(outdir+'/datafiles/'+file_prefix+'_p'+str(p)+'_n'+str(n)+'.csv', sep=',')

def plotCTD(CTD, outdir, file_prefix, fit_CTD, side, nstrips):
    print('plotting CTDs...')
    
    CTD_params = [[[0.,0.,0.,0.] for i in range(nstrips)]for i in range(nstrips)]
    for p in tqdm(range(nstrips)):
        for n in range(nstrips):
            plt.figure()
            hist, bin_edges, _ = plt.hist(CTD[p][n], bins=np.linspace(-350,350,num=100))
            bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
            if fit_CTD:
                fit_mask = bin_centers < 0.
                p0 = [500., -200., 30.]
                if side=='DC':
                    p0 = [500., 125., 30.]
                    fit_mask = bin_centers > 0.
                try:
                    if np.sum(hist) > 10.:
                        popt, pcov = curve_fit(gauss, bin_centers[fit_mask], hist[fit_mask], p0=p0)
                        CTD_params[p][n] = np.concatenate([popt[1:], np.sqrt(np.diag(pcov))[1:]])
                        plt.text(-100, np.max(hist)*0.75, 'Gaussian Peak = ' + str(round(popt[1],2)) + ' +/- ' + str(round(np.sqrt(np.diag(pcov))[1], 2)) + ' ns\n' \
                            + 'Sigma = ' + str(round(popt[2],2)) + ' +/- ' + str(round(np.sqrt(np.diag(pcov))[2], 2)) + ' ns')
                        plt.plot(bin_centers[fit_mask], gauss(bin_centers[fit_mask], *popt), color='red')
                    else:
                        print('Not enough counts to fit p' +str(p) + ', n' +str(n))
                except:
                    print('Failed to fit the data for p' +str(p) + ', n' +str(n))
            plt.ylabel('Counts')
            plt.xlabel('Collection Time Difference (ns)')
            plt.tight_layout()
            plt.savefig(outdir+'/figures/'+file_prefix+'_p'+str(p)+'_n'+str(n)+'.pdf')
            plt.close()

    if fit_CTD:
        with open(outdir + '/CTD_parameters.txt', 'w') as f:
            f.write('### Parameters of the CTD determined using Gaussian fits.\n')
            f.write('### p strip, n strip, gauss mean, gauss sigma, mean err, sigma err\n')
            for p in range(nstrips):
                for n in range(nstrips):
                    f.write(str(p) + ', ' + str(n) + ', ' + \
                        str(CTD_params[p][n][0]) + ', ' + str(CTD_params[p][n][1]) + ', ' + str(CTD_params[p][n][2]) + ', ' + str(CTD_params[p][n][3]) + '\n')
        
        means = np.array(CTD_params)[:,:,0].flatten()
        plt.figure()
        plt.hist(means, bins = 25, range = (np.mean(means)-(1.5*np.std(means)), np.mean(means)+(1.5*np.std(means))))
        plt.xlabel('Mean Collection Time Difference (ns)')
        plt.ylabel('Pixels')
        plt.tight_layout()
        plt.savefig(outdir+'/'+file_prefix + '_meanCTDhist.pdf')
        plt.close()

        plt.figure()
        plt.imshow(np.array(CTD_params)[:,:,0], origin='lower')
        plt.xlabel('n strip')
        plt.ylabel('p strip')
        plt.colorbar(label='Mean CTD')
        plt.tight_layout()
        plt.savefig(outdir+'/'+file_prefix + '_meanCTDmap.pdf')
        plt.close()
 
    image = np.array([[len(CTD[p][n]) for n in range(nstrips)] for p in range(nstrips)])
    counts = np.sum(image)
    plt.figure()
    plt.imshow(image, origin='lower')
    plt.xlabel('n strip')
    plt.ylabel('p strip')
    plt.colorbar(label='Counts')
    plt.tight_layout()
    plt.savefig(outdir+'/'+file_prefix + '_countmap.pdf')
    plt.close()

    return counts

def main(argv):
    infile=''
    outdir=''
    emin = 0.0
    emax = 2000.
    opts, args = getopt.getopt(argv, "hi:o:s:n:f", ['infile=','emin=','emax=','outdir=', 'side=', 'nstrips='])
    filelist = []
    default_outdir = True
    file_prefix = ''
    fit_CTD=False
    side = None
    nstrips = 64
    for opt, arg in opts:
        if opt == '-h':
            print('python dat2CTD.py -i <inputfile> -o <outputdir> --emin <emin> --emax <emax> -s [AC, DC] -n [37,64] -f')
        elif opt in ('-i', '--infile'):
            filelist.append(arg)
        elif opt in ('-o', '--outdir'):
            file_prefix = arg
            default_outdir = False
        elif opt=='--emin':
            emin=float(arg)
        elif opt=='--emax':
            emax=float(arg)
        elif opt in ('-f', '--fit'):
            fit_CTD = True
        elif opt in ('-s', '--side'):
            if arg!='AC' and arg!='DC':
                assert False, "incorrect value for side. Must be AC or DC"
            else:
                side = arg
        elif opt in ('-n', '--nstrips'):
            if int(arg)!=37 and int(arg)!=64:
                assert False, "incorrect value for nstrips. Must be 37 or 64"
            else:
                nstrips = int(arg)
        else:
            assert False, "incorrect option"
    if side is None:
        assert False, "Please indicate the irradiated side of the detector."
    if default_outdir:
        for file in filelist:
            file_prefix = file_prefix + file.split('/')[-1].split('.')[0]
        file_prefix = file_prefix+ '_'+ str(int(emin))+'keV_to_' + str(int(emax))+'keV'
    outdir = '/Users/Sophie/Desktop/COSI/calibrations/Data/CTD_params/' + file_prefix
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        os.mkdir(outdir+'/figures')
        os.mkdir(outdir+'/datafiles')
    CTD = dat2CTD(filelist, emin, emax, nstrips)
    writeCTD(CTD, outdir, file_prefix, nstrips)
    counts = plotCTD(CTD, outdir, file_prefix, fit_CTD, side, nstrips)
    with open(outdir + '/README.txt', 'w') as f:
        f.write('Input files:\n')
        for infile in filelist:
            f.write(infile+'\n')
        f.write('Energy range: ' + str(emin) + ' keV to ' + str(emax) + ' keV\n')
        f.write('Total counts: ' + str(counts))



if __name__=="__main__":
    main(sys.argv[1:])