#!/usr/bin/env python
# coding: utf-8


#code to help read in COSI data from 37 strip detector and create a df with eenrgies and positional information of events 

#import libraries 
import numpy as np
import pandas as pd


#define function which reads in DAT files, cuts multi pixel events, and saves energy and positonal information in df

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