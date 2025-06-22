#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 14:41:25 2025

@author: dani
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal, integrate
from scipy.stats import norm
import pandas as pd
import requests
import os, glob

ds = np.DataSource()

def smooth(data, win=11): 
    smoothed_data = signal.savgol_filter(data, win, 3)
    return smoothed_data

def load_data(shot_no, position, identifier):
    ''' Load plasma position obtained from Mirnov Coils and Fast Cameras '''
    
    try:
        url_mc = f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/LimiterMirnovCoils/plasma_position.csv'
        mc_pos  = pd.read_csv(ds.open(url_mc), index_col = 'Time', usecols = ['Time', identifier]).squeeze()
    except:
        print(f'Missing MC data on the {position} position')
        mc_pos = pd.Series([np.nan])
        
    try:
        url_cam    = f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/FastCameras/Camera_{position}/Camera{position}Position'
        camera_pos = pd.read_csv(ds.open(url_cam), names=['Time', identifier], index_col = 'Time').squeeze()
    except:
        print(f'Missing camera data on the {position} position')
        camera_pos = pd.Series([np.nan])
    
    return mc_pos, camera_pos

def comparison_plot(mc_pos, cam_pos, identifier):
    
    fig = plt.figure(figsize=(12,5),dpi=70)
    
    ax = mc_pos.plot(label='Mirnov coils', color = 'tab:blue')
    ax = cam_pos.plot(label = 'Fast Camera', color = 'tab:red')

    ax.set_ylabel(fr'$\Delta$ {identifier} [mm]', fontsize = 14)
    ax.set_xlabel('Time [ms]', fontsize = 14)
    ax.set_ylim(-85,85) 

    leg = plt.legend(loc = 'best', shadow = False, fancybox=True, edgecolor = 'k')

    for text in leg.get_texts():
          plt.setp(text, fontsize = 12)

    ax.grid(which = 'major', c = 'gray', linewidth = 0.5, linestyle = 'solid') 
    ax.axhline(y=0, color='k', ls='--', lw=1, alpha=0.4)
    # fig.savefig(f'Camera_vs_Mirnov_{identifier}')
    return fig
# %%
# Plot functions
def plot_probability(position, prior, posterior, P_mc, P_fc, plot_all = False):
    # Plot the prior, likelihoods, and posterior distribution
    if plot_all:
        plt.plot(position, prior, label='Prior')
        plt.plot(position, P_fc, label=r'$P_{fc}$')
        plt.plot(position, P_mc, label=r'$P_{mc}$')
    plt.plot(position, posterior, label='Posterior')
    plt.xlabel('Position')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Bayesian Update for Plasma Position Estimation')

def plot_updk(time, cam_pos, mc_pos, updk, sigma_cam, sigma_mc, sigma_updk, identifier, sat,t_plasma_start,t_plasma_end):
    ''' 
    Comparison plot of MC's position, camera's position and estimated position
    '''
    idx_cam = np.where(np.logical_and(cam_pos.index<=time[-1],cam_pos.index>=time[0]))
    
    # maximum plasma radius
    a = 85 
    # Define the position range
    positions = np.linspace(-a, a, 1000)
    
    
    fig, ax = plt.subplots(figsize = (12, 5), dpi=70)
    
    # Position from  Mirnov coils
    ax.plot(time, mc_pos, label = 'Mirnov coils', color='tab:blue')
    ax.fill_between(time, mc_pos - sigma_mc, mc_pos + sigma_mc, color='tab:blue', alpha = 0.3)
    
    # Position from fast camera
    cam_pos.plot(ax = ax, label = 'Fast camera', color='tab:red')
    ax.fill_between(cam_pos.index[idx_cam], cam_pos.values[idx_cam] - sigma_cam, cam_pos.values[idx_cam] + sigma_cam, color='tab:red', alpha = 0.3)
    
    # Estimated position
    ax.plot(time, updk, label = 'Combined position estimate', color = 'tab:green')
    ax.fill_between(time, updk - sigma_updk, updk + sigma_updk, color = 'tab:green', alpha = 0.3)
    
    ax.set_ylabel(fr'$\Delta$ {identifier} [mm]', fontsize = 14)
    ax.set_xlabel('Time [ms]', fontsize = 14)
    ax.set_ylim([-a,a])
    
    ax.set_xlim(t_plasma_start*0.9, t_plasma_end*1.1)
    
    leg = ax.legend(loc = 'best', shadow = False, fancybox=True, edgecolor = 'k')
    for text in leg.get_texts():
          plt.setp(text, fontsize = 12)
    ax.grid(which = 'major', c = 'gray', linewidth = 0.5, linestyle = 'solid') 
    ax.axhline(y=0, color='k', ls='--', lw=1, alpha=0.4)
    
#     if sat.size>1:
#         t_sat = round(sat.index[0], 2)
#         ax.axvline(t_sat, c = 'tab:brown')
#         ax.text(t_sat*0.98,-a*0.95,f'Iron core saturation',rotation=90)
    
    ax.axvline(t_plasma_end, c = 'tab:brown')
    # ax.text(t_plasma_end*0.98,-a*0.95,f'Detected plasma termination',rotation=90)
    
    return fig
    
    # fig.savefig(f'comparison/noIron/Estimated_position_{identifier}_{shot_no}_noIronSat')
    
    
    
def plot_stab(df_stab,t_plasma_start,t_plasma_end):
    fig, axs = plt.subplots(2, 1, figsize = (12, 5), dpi=70, sharex=True)
        
    for ax, identifier in zip(axs.flatten(), ['vertical', 'radial']):
        
        df_stab[identifier].plot(ax = ax, label = f'{identifier} stab')
        
        leg = ax.legend(loc = 'best', shadow = False, fancybox=True, edgecolor = 'k')
        for text in leg.get_texts():
              plt.setp(text, fontsize = 12)
      
        ax.grid(which = 'major', c = 'gray', linewidth = 0.5, linestyle = 'solid') 
        ax.axhline(y=0, color='k', ls='--', lw=1, alpha=0.4)
        
        ax.set_ylabel(fr'$I$ [A]', fontsize = 14)
        ax.set_xlabel('Time [ms]', fontsize = 14)
        
        ax.set_xlim(t_plasma_start*0.9, t_plasma_end*1.1)
        
# %%
def P_diag(position_range, mu_k, sigma_k):
    ''' 
    Likelihood function for diagnostic
    '''
    return norm.pdf(position_range, loc=mu_k, scale=sigma_k) 

def bayesian_update(prior, P_fc, P_mc):
    ''' 
    Bayes' theorem: Update the prior distribution based on the likelihoods 
    '''
    posterior = P_fc * P_mc *prior
    posterior /= np.sum(posterior)  # Normalize to obtain a proper probability distribution
    return posterior

def estimate_postition(positions, mc_pos, cam_pos, sat, stab_switch, sigma_mc_init = 10, sigma_cam_init = 5, plot_posterior = False):
    '''
    Use Bayesian updating to combine the positions obtained from the fast camera and Mirnov coils for more accurate position estimation.
    ''' 
    f_interp = interpolate.interp1d(mc_pos.index,mc_pos.values)

    # Interpolate camera's position to mc's time
    t_s       = mc_pos.index[0] 
    t_e       = min(cam_pos.index[-1], mc_pos.index[-1]) 
    t_new     = cam_pos.index[np.logical_and(cam_pos.index>t_s,cam_pos.index<t_e)]
    mc_interp = f_interp(t_new)
    nt = t_new.size
    nx = positions.size
    
    # Treat cases where there is a large difference between the end times of the discharge
    # Cameras detected the end of the plasma later
    t_diff = cam_pos.index[-1] - mc_pos.index[-1]
    if t_diff > 1:
        pos_res = cam_pos.loc[cam_pos.index>t_e].values
        t_new = np.append(t_new, cam_pos.index[cam_pos.index>t_e])
        n_res = pos_res.size
        mc_interp = np.append(mc_interp, np.full((n_res,1),np.nan))  
    # Mirnov coils detected the end of the plasma later                          
    elif t_diff < -1:
        pos_res = mc_pos.loc[mc_pos.index>t_e].values
        t_new = np.append(t_new, mc_pos.index[mc_pos.index>t_e])
        n_res = pos_res.size
    else:
        n_res = 0
     
    # Create array for storing outputs
    updk = np.zeros(nt+n_res); prior = np.zeros((nt,nx)); posterior = np.zeros((nt,nx)); sigma_updk = np.zeros(nt)
    
    # Initial guess of the standarad deviation
    sigma_updk[0] = 2;
    
    # Standard deviation of diagnostic measurements
    sigma_mc  = np.ones(nt+n_res)*sigma_mc_init
    sigma_cam = np.ones(nt+n_res)*sigma_cam_init
    # Account for the factors affecting MC measurements
    # NOTE: A more sophisticated solution for accounting for these (e.g. with respect to the coil current size + its time evolution) is expected in the (near) future.
    #       A more precise selection of error values is also the subject of future work. 
    if 'on' in stab_switch:
        sigma_mc += 5
    if sat.size > 1:    
        sigma_mc[t_new > sat.index[0]] += 5
    
    if plot_posterior: fig, ax = plt.subplots()
    
    for it in range(0,nt-1):
        # Prior distribution (initial belief about the position)
        prior[it,:] = norm.pdf(positions, loc=updk[it], scale=sigma_updk[it])
        
        P_mc = P_diag(positions, mc_interp[it], sigma_mc[it])
        P_fc = P_diag(positions, cam_pos.values[it], sigma_cam[it])

        # Combine information from both diagnostics
        posterior[it,:] = bayesian_update(prior[it,:], P_mc, P_fc)
        
        # Check for nan
        if np.isnan(posterior[it,:].std()):
            posterior[it,:] = posterior[it-1,:] 
        
        # Estimate the position using maximum a posteriori estimation
        updk[it+1] = positions[np.argmax(posterior[it,:])]
        sigma_updk[it+1] = max(1, abs(positions[posterior[it,:].std()<= posterior[it,:]][0]-updk[it+1]))
        
        
        # Plot posterior probability
        if plot_posterior: plot_probability(positions, prior[it,:], posterior[it,:], P_mc, P_fc)
    
    # treat cases where there is a large difference between the end times of the discharge -> use available data
    if abs(t_diff) > 1:
        updk[nt:nt+n_res] = pos_res
        sigma_updk = np.append(sigma_updk, np.ones(n_res)*sigma_updk[-1]) # use last value -> may be modified in the future
        
    return t_new, updk, mc_interp, sigma_updk, sigma_mc, sigma_cam, prior, posterior


# %% 
def compare_fc_mc(shot_no, mode = 'basic'):
    
    # load data    
    mc_pos_r, cam_pos_r  = load_data(shot_no, 'Radial', 'r')
    mc_pos_z, cam_pos_z  = load_data(shot_no, 'Vertical', 'z')
    
    t_plasma_start = float(requests.get(f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/Results/t_plasma_start').text)
    t_plasma_end = float(requests.get(f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/PlasmaDetection/Results/t_plasma_end').text)
    
    if 'basic' in mode:
        fig_r = comparison_plot(mc_pos_r, cam_pos_r, 'r')
        fig_z = comparison_plot(mc_pos_z, cam_pos_z, 'z')
        
    
    elif 'advance' in mode:
        # Get status on the usage of plasma position stabilization
        # NOTE: A more sophisticated solution for accounting for these is expected in the (near) future -> to make it more robust
        stab_switch = requests.get(f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Infrastructure/PositionStabilization/Parameters/main_switch').text
        print('Stabilization status: ', stab_switch)
        
        
        stab_url     = lambda position: f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Infrastructure/PositionStabilization/U^{position}_currclamp.csv' 
        
        # Currents in the stabilization coils
        cI = 1/0.05 # calibration constant
        df_stab = pd.concat([pd.read_csv(stab_url(pos), names = [pos])*cI for pos in ['radial','vertical']], axis = 'columns')
        # get oscilloscope's time
        dt = float(requests.get(f'http://golem.fjfi.cvut.cz/shotdir/{shot_no}/Devices/Oscilloscopes/RigolMSO5204-a/ScopeSetup/XINC').text)*1e6
        df_stab.index = pd.Series(np.linspace(0, dt, df_stab.shape[0])).rename('Time')
        # smooth signals
        df_stab['radial']   = smooth(df_stab['radial'])
        df_stab['vertical'] = smooth(df_stab['vertical'])
        
        # maximum plasma radius
        a = 85 
        # Define the position range
        positions = np.linspace(-a, a, 1000)
        
        sigma_mc_r_init = 10; sigma_mc_z_init = 8
        sigma_fc_r_init = 5;  sigma_fc_z_init = 5; 
        
        Uloop = pd.read_csv(f'http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/BasicDiagnostics/Results/U_loop.csv', 
                            names = ['Time', 'Uloop'], index_col='Time').squeeze()
        
        phi =  pd.Series(integrate.cumtrapz(Uloop.values, x = Uloop.index*1e-3, initial=0), index=Uloop.index, name = 'phi')
        sat = phi[phi>=0.12]
        
        # Estimate position
        t_r, updk_r, mc_int_r, sigma_updk_r, sigma_mc_r, sigma_cam_r, _, _  = estimate_postition(positions, mc_pos_r, cam_pos_r, sat, stab_switch, sigma_mc_r_init, sigma_fc_r_init)
        t_z, updk_z, mc_int_z, sigma_updk_z, sigma_mc_z, sigma_cam_z, _, _  = estimate_postition(positions, mc_pos_z, cam_pos_z, sat, stab_switch, sigma_mc_z_init, sigma_fc_z_init)

        fig_r = plot_updk(t_r, cam_pos_r, mc_int_r, updk_r, sigma_cam_r, sigma_mc_r, sigma_updk_r, 'r', sat,t_plasma_start,t_plasma_end)
        fig_z = plot_updk(t_z, cam_pos_z, mc_int_z, updk_z, sigma_cam_z, sigma_mc_z, sigma_updk_z, 'z', sat,t_plasma_start,t_plasma_end)

    return fig_r, fig_z

 

    