# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:52:08 2020

@author: klafe
"""
# testing the success of each method
# general stuff goes in the first block, then each method in the block below

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import constants as const
import math

# Define constants
h = const.h
c = const.c
k = const.k

# Defining all the fuctions to be used in the code
def readin(filename, sizez, sizey, sizex):
    # Reads in any fit file into a cube shape
    data = np.zeros((sizez, sizey, sizex))
    with fits.open(filename) as f_hdul:
        data[:, :, :] = f_hdul[0].data
    return data

# Functions for dealing with this specific data set
def resample(m_wave, m_rad, wave_data):
    # Re-sampling modtran data to fit the length of the provided data
    f = interp1d(m_wave, m_rad, fill_value="extrapolate")
    resampled_wave = f(wave_data)
    return resampled_wave
 
def readin_sat(loc, ysize):
    # Read in the sat_mask, which is made to exclude specific points which saturate
    sat_mask = readin(loc + 'sat_mask_regions.fit', 1, ysize, 64)
    sat_mask = np.reshape(sat_mask, (ysize, 64))
    return sat_mask

def ignore_regions(latlong, spectra_data, moon_left, moon_right, sat, incidence):
    lat = latlong[1, :, :]
    mask = lat > -5
    n_spec = spectra_data[34, :, :]
    n_spec[mask] = np.nan    
    long = latlong[0, :, :]
    mask = long < 0
    n_spec[mask] = np.nan
    t = np.zeros((ysize, 64))
    t[:, 0:moon_left] = 1
    t[:, moon_right:64] = 1
    mask = t == 1
    n_spec[mask] = np.nan  
    mask = sat == 0
    n_spec[mask] = np.nan
    regions = n_spec
    mask = incidence < 0
    regions[mask] = np.nan
    
    return regions

# Functions for modelling
def planck(wavelength, temperature, emiss):
    # Thermal part
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-emiss)*top 
    return them

def scatter(wavelength, m, constant):
    # Scattered part
    return m * wavelength + constant

# Functions for integrated band depth
def smooth(yval, box_size):
    # Used to smooth the data 
    box = np.ones(box_size) / box_size
    y_smooth = np.convolve(yval, box, mode='same')
    return y_smooth

def abstrapz(y, x=None, dx=1.0):
    # This is used instead of np.trapz to find the ibd 
    # (ignores positive region totally)
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        d = np.diff(x)
    ret = (d * (y[1:] +y[:-1]) / 2.0)
    return ret[ret>0].sum()  #The important line

# Analysis function
def finding_ibanddepth(wave, app_ref, y, x, st):
    # Integrated using bounds provided. min+max gives region of integration
    if st == '2':
        fit_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
        fit_b = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.59, wave[:, y, x] <= 1.61))))
        fit_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.59, wave[:, y, x] <= 2.61))))
        fit_d = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
        min_bd = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.50, wave[:, y, x] <= 1.51))))
        max_bd = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 2.50, wave[:, y, x] <= 2.52))))
    if st == '3':
        fit_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.59, wave[:, y, x] <= 2.61))))
        fit_b = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
        fit_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.59, wave[:, y, x] <= 3.61))))
        fit_d = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.69, wave[:, y, x] <= 3.71))))
        min_bd = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.59, wave[:, y, x] <= 2.61))))
        max_bd = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 3.54, wave[:, y, x] <= 3.56))))
        
    wave_app = np.concatenate([wave[fit_a:fit_b, y, x], wave[fit_c:fit_d, y, x]])
    spec_app = np.concatenate([app_ref[fit_a:fit_b, y, x], app_ref[fit_c:fit_d, y, x]])
    index_app = ~(np.isnan(wave_app) | np.isnan(spec_app))
    
    # Least squares fit for linear relationship of apparent reflectance
    opt, cov = curve_fit(scatter, wave_app[index_app], spec_app[index_app])
    app_fit = scatter(wave[:, y, x], opt[0], opt[1]) 
    #print(opt[0], opt[1])
    
    # plot the fit
    if y == 60 and x == 40 or y == 40 and x == 30:
        plt.plot(wave[:, y, x], app_fit, label='fit')
        plt.plot(wave[:, y, x], app_ref[:, y, x], label='data')
        plt.legend()
        plt.ylim((0, 0.5))
        plt.xlim((1.0, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.show()
    
    # Define continuum removed reflectance to be the ratio of the actual to the line of best fit
    cm_rem = app_ref[:, y, x]/app_fit
    banddepth = 1 - cm_rem
    
    # Determining the integrated value
    intbanddepth_smooth = np.zeros((512))
    intbanddepth_smooth[:] = np.nan
    
    # Integrate
    length_wave = max_bd - min_bd
    valid = ~np.isnan(banddepth)
    intbanddepth_smooth[valid] = smooth(banddepth[valid], 3)
    valid = ~np.isnan(intbanddepth_smooth[min_bd:max_bd])
    ibd = abstrapz(intbanddepth_smooth[min_bd:max_bd][valid]) / length_wave    

    return banddepth, ibd

# File names/locations, changes per cube
loc = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/fall/research/calibrated_cubes/day339/01/'
wave_filename = loc + 'wave_data_day339_scale001.fit'
spec_filename = loc + 'spat_data_day339_scale001.fit'
latlong_loc = 'C:\\Users/klafe/Desktop/spring_senior/research/pix_to_longlat_Dec05_1.fit'
cosi_filename = 'C:\\Users/klafe/Desktop/spring_senior/research/incidence_ang_Dec05_1.fit'
sat_mask_filename = loc+'sat_mask_regions.fit'
modtran_filename = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/spring/research/Modtran_1nm_4.85um.txt'

date_and_scan = np.loadtxt('C:\\Users/klafe/Desktop/spring_senior/research/date_and_scan_bounds.txt', delimiter='\t')
st = 33901
row = np.where(date_and_scan[0, :] == st)
ysize = 100                                      # size of y-array
moon_left = np.int(date_and_scan[1, row])
moon_right = np.int(date_and_scan[2, row])
wl = 512                                    # length of wavelength array

# Read in relevant additions for limiting data and analysis
latlong = readin(latlong_loc, 2, ysize, 64)
sat_mask = readin_sat(loc, ysize)

# Read in the data
im = readin(spec_filename, wl, ysize, 64)
wave = readin(wave_filename, wl, ysize, 64)

# Read in and define modtran stuff
modtran_wave, modtran_rad = np.loadtxt(modtran_filename, skiprows=1, unpack=True)
modtran = resample(modtran_wave, modtran_rad, wave)
mod = modtran

# Read in the incidence angle affect, in degrees.
cos_ia = np.zeros([ysize, 64])
with fits.open(cosi_filename) as hdul:
    cos_ia[:, :] = hdul[0].data
cos_i = np.cos(np.radians(cos_ia))

# Define guesses - minimal to zero changes when these are changed
temp_guess = 360
m_guess = 0.01
const_guess =  0.02
e_guess = 0.9 #1 - 0.9

# Sets limits on region for anaylsis, excludes saturated and off moon. 
n_spec = ignore_regions(latlong, im, moon_left, moon_right, sat_mask, cos_i)

wavelength_lab, sample1, sample2, sample3, sample4 = np.loadtxt('C:\\Users/klafe\Desktop/spring_senior/research/Lunar_Soils_Lab.txt', skiprows=1, unpack=True)
wavelength_lab = wavelength_lab/1000

#%%
# 1. the simplest: T, e as variables. 

def model_fitting_1(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    # Crop wave and im for scatter fit, ignore any NaN values
    im_adj_2 = im[:, y, x]*np.pi/(modtran[:, y, x]*cosi)
    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x] - scat_part_2[min_c:max_c]*modtran[min_c:max_c, y, x]*cosi/np.pi
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess, e_guess], bounds=[[200, 0.5], [425, 1]])

    thermal_part_2 = planck(wave[:, y, x], opt_t[0], opt_t[1])
    temp_2 = opt_t[0]
    emiss_2 = opt_t[1]
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_2
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
     
    # Determine apparent reflectance
    app_ref = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots             
    if y == 60 and x == 40 or y == 40 and x == 30:
        print('Emissivity: ', emiss_2)
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], thermal_part_2, label='Thermal')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, 3))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*cosi*modtran[:, y, x]/np.pi, label='Scatter fit 2')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part_2, label='fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_2, label='scatter')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    params = np.array([temp_2, m_2, con_2, emiss_2])
    return params, app_ref

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
# how do i do this
y = 60
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x] = model_fitting_1(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print(ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b] = model_fitting_1(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print(ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show()


#%%
# 2. what do i want to do? what is the next logical step? hold emiss = 0.87?
def planck_2(wavelength, temperature):
    # Thermal part
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-0.1)*top 
    return them

def model_fitting_2(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    # Crop wave and im for scatter fit, ignore any NaN values
    im_adj_2 = im[:, y, x]*np.pi/(modtran[:, y, x]*cosi)
    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x] - scat_part_2[min_c:max_c]*modtran[min_c:max_c, y, x]*cosi/np.pi
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_2, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_2 = planck_2(wave[:, y, x], opt_t[0])
    temp_2 = opt_t[0]
    emiss_2 = 0.87
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_2
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
     
    # Determine apparent reflectance
    app_ref = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots             
    if y == 60 and x == 40 or y == 40 and x == 30:
        print('Emissivity: ', emiss_2)
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], thermal_part_2, label='Thermal')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, 3))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*cosi*modtran[:, y, x]/np.pi, label='Scatter fit 2')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part_2, label='fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_2, label='scat')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    params = np.array([temp_2, m_2, con_2, emiss_2])
    return params, app_ref

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
# how do i do this
y = 60
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x] = model_fitting_2(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print(ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b] = model_fitting_2(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print(ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 

#%%
# 3. whats the next step? the two thermal iteration?
def planck_2(wavelength, temperature):
    # Thermal part
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-0.9)*top 
    return them

def model_fitting_3(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    
    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_2, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_2(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    emiss_1 = 0.9
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_1
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # refit thermal, with scatter removed, emissivity as parameters
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    opt, cov = curve_fit(planck, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1, emiss_1])
    temp_2 = opt[0]
    emiss_2 = opt[1]
    thermal_part_2 = planck(wave[:, y, x], temp_2, emiss_2)
    
    # Determine apparent reflectance
    app_ref = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots             
    if y == 60 and x == 40 or y == 40 and x == 30:
        print('Emissivity: ', emiss_2)
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], thermal_part_2, label='Thermal')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, 3))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*cosi*modtran[:, y, x]/np.pi, label='Scatter fit 2')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part_2, label='fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_2, label='Scatter')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
        
    params = np.array([temp_2, m_2, con_2, emiss_2])
    return params, app_ref

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
# how do i do this
y = 50
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x] = model_fitting_3(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b] = model_fitting_3(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 

#%%
# 4. whats the next step? a nother set of iteration?

def planck_2(wavelength, temperature):
    # Thermal part
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-0.9)*top 
    return them

def model_fitting_4(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_2, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_2(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    emiss_1 = 0.9
    
    # Remove thermal component
    im_adj_0 = im[:, y, x] - thermal_part_1
    im_adj_0 = np.pi*im_adj_0/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_0[min_a:max_a], im_adj_0[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # refit thermal, with scatter removed, emissivity as parameters
    im_adj_1 = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    opt, cov = curve_fit(planck, wave[min_c:max_c, y, x][index_therm_2], im_adj_1[min_c:max_c][index_therm_2], bounds=[[200, 0],[450, 1]]) #, p0=[temp_1, emiss_1])
    temp_2 = opt[0]
    emiss_2 = opt[1]
    thermal_part_2 = planck(wave[:, y, x], temp_2, emiss_2)
    
    # redo scatter # this doesnt effect outcome?, so how is this different than method 3?
    im_adj_2 = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    opt, cov = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2]) #, p0=[m_2, con_2])
    scat_part_3 = scatter(wave[:, y, x], opt[0], opt[1])
    
    # re do thermal fit
    im_adj_3 = im[:, y, x] - scat_part_3*modtran[:, y, x]*cosi/np.pi    
    opt, cov = curve_fit(planck, wave[min_c:max_c, y, x][index_therm_2], im_adj_3[min_c:max_c][index_therm_2], bounds=[[200, 0], [450, 1]]) #, p0=[temp_2, emiss_2])
    thermal_part_3 = planck(wave[:, y, x], opt[0], opt[1])
    emiss_2 = opt[1]
    temp_2 = opt[0]
    
    # Determine apparent reflectance
    app_ref = (im[:, y, x] - thermal_part_3)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots             
    if y == 60 and x == 40 or y == 40 and x == 30:
        print('Emissivity: ', emiss_2)
        print('Temp: ', temp_2) 
            
        plt.plot(wave[:, y, x], im_adj_1, label='Data1')
        plt.plot(wave[:, y, x], im_adj_3, label='Data2')
        plt.plot(wave[:, y, x], thermal_part_2, label='Thermal')
        plt.plot(wave[:, y, x], thermal_part_3, label='2nd iteration. ')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, np.nanmean((im_adj_1[max_c-2:max_c+2]))+1.5))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im_adj_0, label='Data1')
        plt.plot(wave[:, y, x], im_adj_2, label='Data2')
        plt.plot(wave[:, y, x], scat_part_2, label='Scatter fit 2')        
        plt.plot(wave[:, y, x], scat_part_3, label='Scatter fit 3')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 3.0))
        plt.ylim((0, np.nanmean((im_adj_0[max_b-2: max_b+2]+0.5))))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part_2, label='fit')
        plt.plot(wave[:, y, x], scat_part_3*modtran[:, y, x]*cosi/np.pi + thermal_part_3, label='improved? fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        plt.plot(wave[:, y, x], thermal_part_3*np.pi/(modtran[:, y, x]*cosi)+scat_part_3, label='improved? fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_3, label='scatter')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    params = np.array([temp_2, m_2, con_2, emiss_2])
    return params, app_ref

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
# how do i do this
y = 60
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x] = model_fitting_4(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b] = model_fitting_4(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 

#%%
for y in range(0, ysize):
    for x in range(0, 64):
        if np.isnan(n_spec[y, x]) == False:
            parameters[:, y, x], apparent_refl[:, y, x], emiss_arr[:, y, x] = model_fitting_14(wave, im, modtran, cos_i, y, x)
            banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
#%%
plt.imshow(parameters[0, :, :], vmin=270, vmax=370)
plt.colorbar()
plt.show()

plt.imshow(parameters[3, :, :], vmin=0.5, vmax = 1.0)#, vmin=270, vmax=370)
plt.colorbar()
plt.show()

plt.imshow(ibd, vmin=0, vmax=0.08)
plt.colorbar()

plt.show()
#%%
# 5. whats next? emissivity as 1-scat? wavelength dep

def model_fitting_5(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    
    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_2, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_2(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_1
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # refit thermal, with scatter removed, emissivity as parameters
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    emiss_a = 1-scat_part_2
    emiss_fit = emiss_a[min_c:max_c][index_therm_2]
    
    def planck_5(wavelength, temperature):
        # Thermal part
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        them = (1-emiss_fit)*top 
        return them

    opt, cov = curve_fit(planck_5, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1])
    temp_2 = opt[0]
    emiss_fit = emiss_a
    emiss_2 = emiss_a
    thermal_part_2 = planck(wave[:, y, x], temp_2, emiss_2) # this does output the same thing as planck_5(wave, temp_2), dont worry. 

    # Determine apparent reflectance
    app_ref = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots  
    if y == 60 and x == 40 or y == 40 and x == 30:           
        print('Emissivity: ', np.nanmean(emiss_2))
        print('Emissivity at 3.5 microns: ', emiss_2[min_c])
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], thermal_part_2, label='Thermal')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, 3))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*cosi*modtran[:, y, x]/np.pi, label='Scatter fit 2')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part_2, label='fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_2, label='scatter')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    params = np.array([temp_2, m_2, con_2, np.nanmean(emiss_2)])
    return params, app_ref, emiss_2

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
emiss_arr = np.zeros((wl, ysize, 64))
# how do i do this
y = 60
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x], emiss_arr[:, y, x] = model_fitting_5(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b], emiss_arr[:, y_b, x_b] = model_fitting_5(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 


#%%
#6. next: emissivity as average of 1-scat at 3.5 or something?
def model_fitting_6(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    
    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_2, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_2(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_1
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # refit thermal, with scatter removed, emissivity as parameters
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    emiss_fit = np.nanmean(1-scat_part_2[max_c-3:max_c+3])
    
    def planck_6(wavelength, temperature):
        # Thermal part
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        them = (1-emiss_fit)*top 
        return them

    opt, cov = curve_fit(planck_6, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1])
    temp_2 = opt[0]
    emiss_2 = emiss_fit
    thermal_part_2 = planck_6(wave[:, y, x], temp_2)

    # Determine apparent reflectance
    app_ref = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots     
    if y == 60 and x == 40 or y == 40 and x == 30:        
        print('Emissivity: ', emiss_2)
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], thermal_part_2, label='Thermal')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, 3))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*cosi*modtran[:, y, x]/np.pi, label='Scatter fit 2')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part_2, label='fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_2, label='scatter')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    params = np.array([temp_2, m_2, con_2, emiss_2])
    return params, app_ref

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
# how do i do this
y = 60
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x] = model_fitting_6(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b]= model_fitting_6(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 

#%%
#7. next: emissivity as wavelength an average of  1-scat
def model_fitting_7(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 3.99, wave[:, y, x] <= 4.01))))
    
    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_2, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_2(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_1
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # refit thermal, with scatter removed, emissivity as parameters
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    emiss_fit = np.nanmean(1-scat_part_2)

    def planck_7(wavelength, temperature):
        # Thermal part
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        them = (1-emiss_fit)*top 
        return them

    opt, cov = curve_fit(planck_7, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1])
    temp_2 = opt[0]
    emiss_2 = emiss_fit
    thermal_part_2 = planck_7(wave[:, y, x], temp_2)

    # Determine apparent reflectance
    app_ref = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots      
    if y == 60 and x == 40 or y == 40 and x == 30:       
        print('Emissivity: ', emiss_2)
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], 1-scat_part_2)
        plt.hlines(emiss_2, 1.0, 5.0, color='black')
        plt.ylim((0.5, 1.0))
        plt.xlim((1.0, 4.8))
        plt.show()
        
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], thermal_part_2, label='Thermal')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, 3))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*cosi*modtran[:, y, x]/np.pi, label='Scatter fit 2')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part_2, label='fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_2, label='scatter')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    params = np.array([temp_2, m_2, con_2, emiss_2])
    return params, app_ref

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
# how do i do this
y = 60
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x] = model_fitting_7(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b] = model_fitting_7(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 

#%% 
#8. next: emissivity as wavelength dept 1-app ref
def planck_2(wavelength, temperature):
    # Thermal part
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-0.9)*top 
    return them

def model_fitting_8(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    
    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_2, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_2(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_1
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # refit thermal, with scatter removed, emissivity as parameters
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    emiss_fita = 1-scat_part_2
    emiss_fit = emiss_fita[min_c:max_c][index_therm_2]
    
    def planck_8(wavelength, temperature):
        # Thermal part
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        them = (1-emiss_fit)*top 
        return them

    opt, cov = curve_fit(planck_8, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1])
    temp_2 = opt[0]
    emiss_fit = emiss_fita
    thermal_part_1 = planck_8(wave[:, y, x], temp_2)

    # Determine apparent reflectance
    app_ref_old = (im[:, y, x] - thermal_part_1)*np.pi/(modtran[:, y, x]*cosi)
    
    # now try refitting
    emiss_2a = 1 - app_ref_old
    emiss_2 = emiss_2a[min_c:max_c][index_therm_2]
    
    def planck_8_2(wavelength, temperature):
        # Thermal part
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        them = (1-emiss_2)*top 
        return them
    
    opt, cov = curve_fit(planck_8_2, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_2])
    emiss_2 = emiss_2a
    temp_2 = opt[0]
    thermal_part_2 = planck_8_2(wave[:, y, x], temp_2)
    app_ref = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots    
    if y == 60 and x == 40 or y == 40 and x == 30:     
        print('Emissivity: ', np.nanmean(emiss_2))
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], 1-scat_part_2)
        plt.plot(wave[:, y, x], emiss_2, 1.0, 5.0, color='black')
        plt.ylim((0.5, 1.0))
        plt.xlim((1.0, 4.8))
        plt.show()
        
        plt.plot(wave[:, y, x], im_adj, label='Data')
        plt.plot(wave[:, y, x], thermal_part_1, label='Thermal')
        plt.plot(wave[:, y, x], thermal_part_2, label='Second thermal fit')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, 3))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*cosi*modtran[:, y, x]/np.pi, label='Scatter fit 2')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part_2, label='fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_2, label='scatter')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    params = np.array([temp_2, m_2, con_2, np.nanmean(emiss_2)])
    return params, app_ref, emiss_2

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
emiss_arr = np.zeros((wl, ysize, 64))
# how do i do this
y = 60
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x], emiss_arr[:, y, x] = model_fitting_8(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b], emiss_arr[:, y_b, x_b] = model_fitting_8(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 


#%%
# 9. next: emissivity as some weird shape/ only matters in thermal so only adjust there
# this is by hand adjusting. 
def planck_2(wavelength, temperature):
    # Thermal part
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-0.9)*top 
    return them

def model_fitting_9(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 3.99, wave[:, y, x] <= 4.01))))
    check = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.99, wave[:, y, x] <= 3.01))))
    ch = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.79, wave[:, y, x] <=2.81))))
    
    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_2, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_2(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    emiss_1 = 0.9
    
    # Remove thermal component
    im_adj_0 = im[:, y, x] - thermal_part_1
    im_adj_0 = np.pi*im_adj_0/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_0[min_a:max_a], im_adj_0[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # refit thermal, with scatter removed, emissivity as parameters
    im_adj_1 = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    opt, cov = curve_fit(planck, wave[min_c:max_c, y, x][index_therm_2], im_adj_1[min_c:max_c][index_therm_2], bounds=[[200, 0],[450, 1]]) #, p0=[temp_1, emiss_1])
    temp_2 = opt[0]
    emiss_2 = opt[1]
    thermal_part_2 = planck(wave[:, y, x], temp_2, emiss_2)
    
    # redo scatter # this doesnt effect outcome?, so how is this different than method 3?
    im_adj_2 = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    opt, cov = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2]) #, p0=[m_2, con_2])
    scat_part_3 = scatter(wave[:, y, x], opt[0], opt[1])
    print(emiss_2)
    
    # re do thermal fit
    im_adj_3 = im[:, y, x] - scat_part_3*modtran[:, y, x]*cosi/np.pi  
    if y == 60 and x == 40:   
        emiss_a = np.ones((512))*emiss_2
        slope = -0.05  # -0.035
        b = emiss_2 - slope*3.0 #*4.0/0.5
        emiss_a[check:min_c] = slope * wave[check:min_c, y, x] + b
        bnew = emiss_a[min_c -1] - slope*3.5/2
        emiss_a[min_c:max_c] = slope*wave[min_c:max_c, y, x]/2 + bnew
        bfin = emiss_a[max_c-1] - slope*4.0/2
        emiss_a[max_c:] = slope*wave[max_c:, y, x] + bfin
        emiss_a[max_c:ch] = slope*wave[max_c:ch, y, x]/2 + bfin
        b_ch = emiss_a[ch-1] - slope*4.3
        emiss_a[ch:] = slope*wave[ch:, y, x] +b_ch
        
        test_emiss = np.loadtxt('C:\\Users/klafe/Desktop/spring_senior/research/emiss_fit_test.csv', delimiter=',')
        emiss_a = resample(test_emiss[0:30, 0], test_emiss[0:30, 1], wave[:, y, x])
        
        emiss_a[min_b:max_c] = 0.8
        emiss_a[max_c:] = 0.78
        
        # beyond max_c is too steep?

    elif y == 40 and x == 30:
        emiss_a = np.ones((512))*emiss_2
        slope = -0.055  #-0.020
        b = emiss_2 - slope*3.0/1.5 #*4.0/0.5
        emiss_a[check:min_c] = slope/1.5 * wave[check:min_c, y, x] + b
        bnew = emiss_a[min_c - 1] - slope*3.5/2
        emiss_a[min_c:max_c] = slope*wave[min_c:max_c, y, x]/2 + bnew
        bfin = emiss_a[max_c-1] -slope*4.0/2
        emiss_a[max_c:ch] = slope*wave[max_c:ch, y, x]/2 + bfin
        b_ch = emiss_a[ch-1] - slope*4.3
        emiss_a[ch:] = slope*wave[ch:, y, x] +b_ch
    
    elif y == 50 and x == 40:
        print('emiss is ', emiss_2)
        emiss_2 = 0.80
        emiss_a = np.ones((512))*emiss_2
        slope = -0.02  #-0.020
        b = emiss_2 - slope*wave[check, y, x]/4 #*4.0/0.5
        emiss_a[check:min_c] = slope/4 * wave[check:min_c, y, x] + b
        
        # only below effects
        slope = -0.006#+0.0005
        bnew = emiss_a[min_c-1] - slope*wave[min_c, y, x]
        emiss_a[min_c:max_c] = slope*wave[min_c:max_c, y, x] +bnew
        slope = -0.06 #-0.060
        bfin = emiss_a[max_c-1] - slope*wave[max_c, y, x]
        emiss_a[max_c:] = slope*wave[max_c:, y, x]+bfin
        
        emiss_a[min_b:min_c] = 0.8 - 0.005*wave[min_b:min_c, y, x]
        print(emiss_a[min_c])
        emiss_a[min_c] =  0.785 #- 0.005*wave[min_c:, y, x] #emiss_a[min_c-1] - 0.005# - 0.005*wave[max_c:, y, x] #0.78
        #emiss_a[max_c:] = 0.775 - slope*wave[max_c:, y, x]
        #emiss_a = 1- resample(wavelength_lab, sample2, wave[:, y, x])
        
        #bnew = 0.8
        #bnew = emiss_a[min_c - 1] - slope*3.5/1
        #emiss_a[min_c:max_c] = slope*wave[min_c:max_c, y, x]/1 + bnew
        
        #emiss_a[min_c:max_c] = 0.80
        #emiss_a[min_c:ch] = 0.81
        #bfin = emiss_a[max_c-1] -slope*4.0/1
        #emiss_a[max_c:ch] = slope*wave[max_c:ch, y, x]/1 + bfin
        #b_ch = emiss_a[ch-1] - slope*4.3
        #emiss_a[ch:] = slope*wave[ch:, y, x] +b_ch


    
    emiss_fit = emiss_a[min_c:max_c][index_therm_2]
    
    def planck_9(wavelength, temperature):
        # Thermal part
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        them = (1-emiss_fit)*top 
        return them
    
    opt, cov = curve_fit(planck_9, wave[min_c:max_c, y, x][index_therm_2], im_adj_3[min_c:max_c][index_therm_2], bounds=[[200], [450]]) #, p0=[temp_2, emiss_2])
    emiss_fit = emiss_a   
    thermal_part_3 = planck_9(wave[:, y, x], opt[0])
    emiss_2 = emiss_a
    temp_2 = opt[0]
    
    # Determine apparent reflectance
    app_ref = (im[:, y, x] - thermal_part_3)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots   
    if y == 50 and x == 40 or y == 40 and x == 30:          
        print('Emissivity: ', np.nanmean(emiss_2))
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], emiss_2, label='emissivity')
        plt.plot(wave[:, y, x], 1 - scat_part_2, label='scat 2')
        plt.plot(wave[:, y, x], 1- scat_part_3, label='scat 3')
        plt.plot(wave[:, y, x], np.average((1-scat_part_2, 1-scat_part_3), axis=0), label='avg')
        title_str = 'Emissivity at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.ylim((0.0, 1.0))
        plt.title(title_str)
        plt.xlim((1.0, 4.85))
        plt.legend()
        plt.show()
        
        plt.plot(wave[:, y, x], emiss_2)
        plt.xscale('log')
        plt.xlim((1, 6))
        plt.show()
        
        plt.plot(wave[:, y, x], im_adj_1, label='Data1')
        plt.plot(wave[:, y, x], im_adj_3, label='Data2')
        plt.plot(wave[:, y, x], thermal_part_2, label='Thermal')
        plt.plot(wave[:, y, x], thermal_part_3, label='2nd iteration. ')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, im_adj_1[max_c]+1.5))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im_adj_0, label='Data1')
        plt.plot(wave[:, y, x], im_adj_2, label='Data2')
        plt.plot(wave[:, y, x], scat_part_2, label='Scatter fit 2')        
        plt.plot(wave[:, y, x], scat_part_3, label='Scatter fit 3')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.5))
        plt.ylim((0, im_adj_0[max_b]+0.5))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part_2, label='fit')
        plt.plot(wave[:, y, x], scat_part_3*modtran[:, y, x]*cosi/np.pi + thermal_part_3, label='improved? fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        plt.plot(wave[:, y, x], thermal_part_3*np.pi/(modtran[:, y, x]*cosi)+scat_part_3, label='improved? fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_2, label='scatter')
        plt.plot(wave[:, y, x], scat_part_3, label='scatter 3')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    params = np.array([temp_2, m_2, con_2, np.nanmean(emiss_2)])
    return params, app_ref, emiss_2

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
emiss_arr = np.zeros((wl, ysize, 64))
# how do i do this
y = 50
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x], emiss_arr[:, y, x] = model_fitting_9(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b], emiss_arr[:, y_b, x_b] = model_fitting_9(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 

#%%

#%%
# 10. combining two temps

# start with simply looking at a single pixel (60, 40), that we have a few estimates of temp for, and use those to show the difference between one and two and three combinded temps
# leave emissivity the same, but maybe vary later

y = 60
x = 40
t1 = 358
t2 = 360
t3 = 350
scat = scatter(wave[:, y, x], parameters[1, y, x], parameters[2, y, x])
temp1 = planck(wave[:, y, x], t1, 0.65)
temp2 = planck(wave[:, y, x], t2, 0.65)
temp3 = planck(wave[:, y, x], t3, 0.56)
temp_avg = planck(wave[:, y, x], np.average((t1, t2, t3)), 0.7)
therm_avg = np.average((temp1, temp2, temp3), axis=0)

plt.plot(wave[:, y, x], im[:, y, x], label='data')
plt.plot(wave[:, y, x], temp_avg+scat*(modtran[:, y, x]*cos_i[y, x])/np.pi, label='avg temp')
plt.plot(wave[:, y, x], therm_avg + scat*(modtran[:, y, x]*cos_i[y, x])/np.pi, label='avg therm')
plt.legend()
plt.ylim((1,4))
plt.xlim((3.0, 4.9))
plt.show()


plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cos_i[y, x]), label='data')
plt.plot(wave[:, y, x], temp_avg*np.pi/(modtran[:, y, x]*cos_i[y, x])+scat, label='avg temp')
plt.plot(wave[:, y, x], therm_avg*np.pi/(modtran[:, y, x]*cos_i[y, x]) + scat, label='avg therm')
plt.legend()
plt.ylim((0.1, 10))
plt.xlim((3.0, 4.8))
plt.yscale('log')
plt.show()

plt.plot(wave[:, y, x], temp_avg-therm_avg)
plt.show()

plt.plot(wave[:, y, x], im[:, y, x], label='data')
plt.plot(wave[:, y, x], temp1+ scat*(modtran[:, y, x]*cos_i[y, x])/np.pi, label='T=%2.0f'%t1, linestyle='dashed')
plt.plot(wave[:, y, x], temp2+ scat*(modtran[:, y, x]*cos_i[y, x])/np.pi, label='T=%2.0f'%t2, linestyle='dotted')
plt.plot(wave[:, y, x], temp3+ scat*(modtran[:, y, x]*cos_i[y, x])/np.pi, label='T=%2.0f'%t3, linestyle='dashdot')
plt.plot(wave[:, y, x], temp_avg+scat*(modtran[:, y, x]*cos_i[y, x])/np.pi, label='avg temp')
plt.plot(wave[:, y, x], therm_avg + scat*(modtran[:, y, x]*cos_i[y, x])/np.pi, label='avg therm')
plt.ylim((1, 4))
plt.xlim((3.0, 4.85))
plt.legend()
plt.show()
#%%
#11. my very first way.
# this will probably be easier to start with then combining two temps
# model just does curvefit
# lets do it the way weve been doing it
def planck_11(temperature, wavelength):
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = top 
    return them

def model_fitting_11(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    # in this version we do both at once using the regions defined
    # start with making a wave and spec array from regions
    wave_cut = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x], wave[min_c:max_c, y, x]])
    spec_cut = np.concatenate([im[min_a:max_a, y, x], im[min_b:max_b, y, x], im[min_c:max_c, y, x]])
    ind = ~np.isnan(spec_cut)
    modtran_cut = np.concatenate([modtran[min_a:max_a, y, x], modtran[min_b:max_b, y, x], modtran[min_c:max_c, y, x]])[ind]
    
    def spectra_model(wavelength, temp, m, constant, e):
        return ((1-e) * planck_11(temp, wavelength)) + scatter(m, wavelength, constant)*modtran_cut*cosi/np.pi

    # then do
    opt, cov = curve_fit(spectra_model, wave_cut[ind], spec_cut[ind], bounds=[[200, 0, -1, -1], [450, 1, 1, 1]])
    temp_2 = opt[0]
    emiss_2 = opt[3]
    m_2 = opt[1]
    con_2 = opt[2]
    
    thermal_part = planck(wave[:, y, x], temp_2, emiss_2)
    scat_part_2 = scatter(wave[:, y, x], m_2, con_2)
    app_ref = (im[:, y, x] - thermal_part)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots that matter right now: apparent reflectance, thermal fit.  
    if y == 60 and x == 40 or y == 40 and x == 30:          
        print('Emissivity: ', emiss_2)
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], im[:, y, x]-scat_part_2*modtran[:, y, x]*cosi/np.pi, label='Data')
        plt.plot(wave[:, y, x], thermal_part, label='Thermal')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, 5))
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_2, label='scatter')
    
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*cosi*modtran[:, y, x]/np.pi, label='Scatter fit 2')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part, label='fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
    
    params = np.array([temp_2, m_2, con_2, emiss_2])

    return params, app_ref

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
# how do i do this
y = 60
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x] = model_fitting_11(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b]= model_fitting_11(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 

#%%
# 12. a test, 6 but with unc to show things

def model_fitting_6(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_2, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_2(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_1
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # refit thermal, with scatter removed, emissivity as parameters
    
    # going to do this three times (avg, +2 sigma, - 2sigma. )
    #avg
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    emiss_fit = np.nanmean(1-scat_part_2[max_c-3:max_c+3])
    
    def planck_6(wavelength, temperature):
        # Thermal part
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        them = (1-emiss_fit)*top 
        return them

    opt, cov = curve_fit(planck_6, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1])
    temp_2 = opt[0]
    emiss_2 = emiss_fit
    thermal_part_2 = planck_6(wave[:, y, x], temp_2)
    app_ref = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)

    # +2
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    emiss_fit = np.nanmean(1-scat_part_2[max_c-3:max_c+3])+2*0.003
    opt, cov = curve_fit(planck_6, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1])
    temp_2_b = opt[0]
    emiss_2_b = emiss_fit
    thermal_part_2_b = planck_6(wave[:, y, x], temp_2)
    app_ref_b = (im[:, y, x] - thermal_part_2_b)*np.pi/(modtran[:, y, x]*cosi)

    # -2
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    emiss_fit = np.nanmean(1-scat_part_2[max_c-3:max_c+3])-2*0.003
    opt, cov = curve_fit(planck_6, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1])
    temp_2_c = opt[0]
    emiss_2_c = emiss_fit
    thermal_part_2_c = planck_6(wave[:, y, x], temp_2)
    app_ref_c = (im[:, y, x] - thermal_part_2_c)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots that matter right now: apparent reflectance, thermal fit.  
    if y == 60 and x == 40 or y == 40 and x == 30:          
        print('Emissivity: ', emiss_2)
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], im_adj, label='Data', c='red')
        plt.plot(wave[:, y, x], thermal_part_2, label='Thermal', c='black')
        plt.fill_between(wave[:, y, x], thermal_part_2, thermal_part_2_b, alpha=0.5, color='blue')
        plt.fill_between(wave[:, y, x], thermal_part_2_c, thermal_part_2, alpha=0.5, color='green')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, im_adj[max_c]+1.5))
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data', c='red')
        plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit', c='black')
        plt.fill_between(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, thermal_part_2_b*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, alpha=0.5, color='blue')
        plt.fill_between(wave[:, y, x], thermal_part_2_c*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, alpha=0.5, color='green')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2', c='black')
        plt.fill_between(wave[:, y, x], app_ref, app_ref_b, alpha=0.5, color='blue')
        plt.fill_between(wave[:, y, x], app_ref_c, app_ref, alpha=0.5, color='green')
        plt.plot(wave[:, y, x], scat_part_2, label='scatter')
    
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*cosi*modtran[:, y, x]/np.pi, label='Scatter fit 2')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part_2, label='fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
    params = np.array([temp_2, m_2, con_2, emiss_2])
    return params, app_ref

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
# how do i do this
y = 60
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x] = model_fitting_6(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b]= model_fitting_6(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 

#%%
#13. using the apparent reflectance as scatter light, then get new thermal and emissivity?
def planck_13(wavelength, temperature):
    # Thermal part
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-0.9)*top 
    return them

def model_fitting_13(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    
    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_2, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_2(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    emiss_1 = 0.9
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_1
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # refit thermal, with scatter removed, emissivity as parameters
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    opt, cov = curve_fit(planck, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1, emiss_1])
    temp_2 = opt[0]
    emiss_2 = opt[1]
    thermal_part_2 = planck(wave[:, y, x], temp_2, emiss_2)
    
    # Determine apparent reflectance
    app_ref = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    
    #now, instead of moving on, lets fit to app_ref, use that as scatter, remove
    # scatter from im, refit thermal, compare difference in temperatures and emiss
    fit_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.59, wave[:, y, x] <= 2.61))))
    fit_b = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    fit_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.59, wave[:, y, x] <= 3.61))))
    fit_d = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.69, wave[:, y, x] <= 3.71))))
    app_ref_fit = np.concatenate([app_ref[fit_a:fit_b], app_ref[fit_c:fit_d]])
    wave_fit = np.concatenate([wave[fit_a:fit_b, y, x], wave[fit_c:fit_d, y, x]])
    ind = ~np.isnan(app_ref_fit)
    opt, cov = curve_fit(scatter, wave_fit[ind], app_ref_fit[ind])
    app_fit = scatter(wave[:, y, x], opt[0], opt[1])    
    
    im_adj_1 = im[:, y, x] - app_fit*modtran[:, y, x]*cosi/np.pi
    opt, cov = curve_fit(planck, wave[min_c:max_c, y, x][index_therm_2], im_adj_1[min_c:max_c][index_therm_2], p0=[temp_1, emiss_1])
    thermal_part_3 = planck(wave[:, y, x], opt[0], opt[1])
    emiss_2 = opt[1]
    temp_2 = opt[0]
    
    app_ref = (im[:, y, x] - thermal_part_3)*np.pi/(modtran[:, y, x]*cosi) 
    
    # then try emiss = 1-app ref
    
    # plots     
    if y == 60 and x == 40 or y == 40 and x == 30:        
        print('Emissivity: ', emiss_2)
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], im_adj, label='Data 1')
        plt.plot(wave[:, y, x], thermal_part_3, label='3rd')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, 3))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*cosi*modtran[:, y, x]/np.pi, label='Scatter fit 2')        
        plt.plot(wave[:, y, x], app_fit*cosi*modtran[:, y, x]/np.pi, label='Scatter fit 3')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + thermal_part_2, label='fit')
        plt.plot(wave[:, y, x], app_fit*modtran[:, y, x]*cosi/np.pi + thermal_part_3, label='fit 3')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        plt.plot(wave[:, y, x], thermal_part_3*np.pi/(modtran[:, y, x]*cosi)+app_fit, label='fit 3')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_2, label='Scatter')
        plt.plot(wave[:, y, x], app_fit, label='Scatter 3')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
        
    params = np.array([temp_2, m_2, con_2, emiss_2])
    return params, app_ref

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
# how do i do this
y = 60
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x] = model_fitting_13(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b] = model_fitting_13(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 

#%%
#14. emissivity fit as a wavelength dependent value
def planck_14(wavelength, temperature):
    # Thermal part
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-0.9)*top 
    return them

def model_fitting_14(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_14, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_14(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    emiss_1 = 0.9
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_1
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # refit thermal, with scatter removed, emissisivity as 1-scat
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    
    # define an emissivty
    emiss_a = 1- scat_part_2
    emiss_fit = 1 - scat_part_2[min_c:max_c][index_therm_2]
    
    # fit to get new temp
    def planck_fit_14(wavelength, temperature):
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        them = (1-emiss_fit)*top
        return them
    
    opt, cov = curve_fit(planck_fit_14, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1])
    temp_2 = opt[0]
    emiss_fit = emiss_a
    therm_part_2 = planck_fit_14(wave[:, y, x], temp_2)
    
    # now, try fitting emissivity as a slope and intercept? keeping temp the same
    
    # if i fit this to a straight line what do i get
    
    opt, cov = curve_fit(scatter, wave[min_c:max_c, y, x][index_therm_2], emiss_fit[min_c:max_c][index_therm_2])
    test_out = scatter(wave[:, y, x], opt[0], opt[1])
    temp_a = temp_2
    
    # if i fit to a im adj in planck fit what do i get
    # define a new planck function, then use it
    def planck_fit(wavelength, slope_check, interc):
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temp_a)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        emissivity = scatter(wavelength*10**(6), slope_check, interc)
        them = (1-emissivity)*top
        return them
        
    
    # fit the values
    # what happens if i remove bounds?
    index_therm_3 = ~np.isnan(im_adj[min_c:max_c])
    opt, cov = curve_fit(planck_fit, wave[min_c:max_c, y, x][index_therm_3], im_adj[min_c:max_c][index_therm_3], bounds=[[-0.1, 0.5],[-0.005, 0.95]])
    emiss_2 = scatter(wave[:, y, x], opt[0], opt[1])# slope_check, opt[0])
    thermal_part_2 = planck(wave[:, y, x], temp_2, emiss_2)
    print(opt[0], opt[1])
    
    def planck_fit(wavelength, slope_c, interc):
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temp_a)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        emissivity = scatter(wavelength*10**(6), slope_c, interc)
        them = (1-emissivity)*top
        return them
    
    # redo scatter?
    im_adj_2 = (im[:, y, x] - thermal_part_2)*np.pi/(modtran[:, y, x]*cosi)
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    opt, cov = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2])
    new_scat = scatter(wave[:, y, x], opt[0], opt[1])
    m_2 = opt[0]
    con_2 = opt[1]
    
    # should i refit the temp?? emiss?? how can i fix this to fix app ref?
    im_adj_3 = im[:, y, x] - new_scat*modtran[:, y, x]*cosi/np.pi
    emiss_fit = emiss_2[min_c:max_c][index_therm_2]
    opt, cov = curve_fit(planck_fit_14, wave_therm_2[index_therm_2], im_adj_3[min_c:max_c][index_therm_2], p0=[temp_2])
    thermal_part_3 = planck(wave[:, y, x], opt[0], emiss_2) 
    emiss_1a = emiss_2
    
    # now refit emissivity?
    index_therm_3 = ~np.isnan(im_adj_3[min_c:max_c])
    temp_a = opt[0]
    opt, cov = curve_fit(planck_fit, wave[min_c:max_c, y, x][index_therm_3], im_adj_3[min_c:max_c][index_therm_3], bounds=[[-0.1, 0.5],[-0.005, 0.95]])#, p0=[0, 0.72], bounds=[[-0.1, 0.5],[0.1, 0.95]])
    emiss_2 = scatter(wave[:, y, x], opt[0], opt[1])
    thermal_part_4 = planck(wave[:, y, x], temp_a, emiss_2)
    temp_4 = temp_a
    
    # Determine apparent reflectance
    app_ref = (im[:, y, x] - thermal_part_4)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots           
    if y == 47 and x == 19 or y == 40 and x == 30:
        print('Emissivity: ', np.nanmean(emiss_2))
        print('Temp: ', temp_4) 
        
        plt.plot(wave[:, y, x], emiss_2, label='final')
        plt.plot(wave[:, y, x], emiss_1a, label='2nd to last')
        plt.plot(wave[:, y, x], emiss_a, label='first')
        plt.legend()
        plt.ylim((0, 1))
        plt.xlim((1.0, 4.8))
        plt.show()   
        
            
        plt.plot(wave[:, y, x], im_adj, label='Data')
        plt.plot(wave[:, y, x], thermal_part_4, label='Fit')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, 4))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], new_scat*cosi*modtran[:, y, x]/np.pi, label='Scatter fit')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
    
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], new_scat*modtran[:, y, x]*cosi/np.pi + thermal_part_4, label='fit')
    
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], thermal_part_4*np.pi/(modtran[:, y, x]*cosi)+new_scat, label='fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], new_scat, label='Scatter')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    params = np.array([temp_4, m_2, con_2, np.nanmean(emiss_2)])
    return params, app_ref, emiss_2

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
emiss_arr = np.zeros((wl, ysize, 64))
# how do i do this
y = 47
x = 19
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x], emiss_arr[:, y, x] = model_fitting_14(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b], emiss_arr[:, y_b, x_b] = model_fitting_14(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 


print(parameters[:, y, x])
print(ibd[y, x])

#%%
s2 = apparent_refl
#%%
plt.plot(wave[:, 70, 15], s0[:, 70, 15])
plt.plot(wave[:, 70, 38], s1[:, 70, 38])
plt.ylim((0, 0.5))
plt.show()

#%%
# need to make the plotting agaisnt each other into a function to make it easier. 
y = 50
x = 40
min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.99, wave[:, y, x] <= 4.01))))
max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))
print(im[min_c, y, x]/im[max_c, y, x])
print(wave[max_c, y ,x])
print(max_c)
#%%
# a test: from that paper
#15. emissivity in planck????
def planck_15(wavelength, temperature):
    # Thermal part
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-0.9)*top 
    return them

def model_fitting_15(wave, im, modtran, incidence, y, x):
    cosi = incidence[y, x]

    # Limits for fits, c for thermal, a+b for scatter. 
    min_a = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 1.49, wave[:, y, x] <= 1.51))))
    max_a = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 1.69, wave[:, y, x] <= 1.71))))
    min_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.49, wave[:, y, x] <= 2.51))))
    max_b = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 2.69, wave[:, y, x] <= 2.71))))
    min_c = np.min(np.argwhere((np.logical_and(wave[:, y, x] >= 3.49, wave[:, y, x] <= 3.51))))
    max_c = np.max(np.argwhere((np.logical_and(wave[:, y, x] >= 4.29, wave[:, y, x] <= 4.31))))

    # Crop wave and im for thermal fit with e=0.9, ignore any nan values.  
    wave_therm_2 = wave[min_c:max_c, y, x]
    spec_therm_2 = im[min_c:max_c, y, x]
    index_therm_2 = ~(np.isnan(wave_therm_2) | np.isnan(spec_therm_2))
    
    # Use least square fit to find best fit temp.
    opt_t, cov_t = curve_fit(planck_15, wave_therm_2[index_therm_2], spec_therm_2[index_therm_2], p0=[temp_guess], bounds=[[200], [425]])

    thermal_part_1 = planck_15(wave[:, y, x], opt_t[0])
    temp_1 = opt_t[0]
    emiss_1 = 0.9
    
    # Remove thermal component
    im_adj_2 = im[:, y, x] - thermal_part_1
    im_adj_2 = np.pi*im_adj_2/(cosi*modtran[:, y, x])
    
    # Crop wave and im for scatter fit, ignore any NaN values    
    wave_scat_2 = np.concatenate([wave[min_a:max_a, y, x], wave[min_b:max_b, y, x]])
    spec_scat_2 = np.concatenate([im_adj_2[min_a:max_a], im_adj_2[min_b:max_b]])  
    index_scat_2 = ~(np.isnan(wave_scat_2) | np.isnan(spec_scat_2))
    
    # Use least square fit to find best slope and intercept for scattered light
    opt_2, cov_2 = curve_fit(scatter, wave_scat_2[index_scat_2], spec_scat_2[index_scat_2], p0=[m_guess, const_guess])
    scat_part_2 = scatter(wave[:, y, x], opt_2[0], opt_2[1])
    m_2 = opt_2[0]
    con_2 = opt_2[1]
    
    # refit thermal, with scatter removed, emissisivity as 1-scat
    im_adj = im[:, y, x] - scat_part_2*modtran[:, y, x]*cosi/np.pi
    
    emiss_fit = np.nanmean(1- scat_part_2)
    
    # fit to get new temp
    def planck_fit_15(wavelength, temperature, emissivity):
        wavelength = wavelength * 10 ** (-6)
        bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
        top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
        top = top*10**(-6)
        them = (1-emissivity)*top
        return them
    
    opt, cov = curve_fit(planck_fit_15, wave[min_c:max_c, y, x][index_therm_2], im_adj[min_c:max_c][index_therm_2], p0=[temp_1, emiss_fit])
    temp_2 = opt[0]
    emiss_2 = opt[1]
    therm_part_2 = planck_fit_15(wave[:, y, x], temp_2, emiss_2)
    
    # Determine apparent reflectance
    app_ref = (im[:, y, x] - therm_part_2)*np.pi/(modtran[:, y, x]*cosi)
    
    # plots           
    if y == 60 and x == 30 or y == 40 and x == 30:
        print('Emissivity: ', np.nanmean(emiss_2))
        print('Temp: ', temp_2) 
        
        plt.plot(wave[:, y, x], emiss_2*np.ones(512), label='final')
        plt.legend()
        plt.ylim((0, 1))
        plt.xlim((1.0, 4.8))
        plt.show()   
        
            
        plt.plot(wave[:, y, x], im_adj, label='Data')
        plt.plot(wave[:, y, x], therm_part_2, label='Fit')
        plt.plot(wave[:, y, x], thermal_part_1, label='og fit')
        plt.legend()
        title_str = 'Thermal fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((3.0, 4.8))
        plt.ylim((0, 4))
        plt.show()
        
        # 2. fitting scatterd
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*cosi*modtran[:, y, x]/np.pi, label='Scatter fit')        
        plt.legend()
        title_str = 'Scatter fit at ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((0.5, 4.8))
        plt.ylim((0, 36))
        plt.show()
    
        
        #3. fit thermal with new emissivity
        plt.plot(wave[:, y, x], im[:, y, x], label='Data')
        plt.plot(wave[:, y, x], scat_part_2*modtran[:, y, x]*cosi/np.pi + therm_part_2, label='fit')
        plt.vlines(4.65, 0, 6)
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0, 6))
        plt.legend()
        plt.show()
        
        #3b. fit thermal with new emissivity like silvias plots
        plt.plot(wave[:, y, x], im[:, y, x]*np.pi/(modtran[:, y, x]*cosi), label='Data')
        plt.plot(wave[:, y, x], therm_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit')
        title_str = 'Full fits ' +'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.xlim((1.0, 4.8))
        plt.ylim((0.1, 10))
        plt.yscale('log')
        plt.legend()
        plt.show()    
        
        # 5. Apparent reflectance (focus on turn off)
        plt.plot(wave[:, y, x], app_ref, label='App ref 2')
        plt.plot(wave[:, y, x], scat_part_2, label='Scatter')
        plt.ylim((0, 0.5))
        plt.xlim((1.2, 4.8))
        title_str = 'Apparent reflectance '+'%2.0f' %y + ',' + '%2.0f' %x
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    params = np.array([temp_2, m_2, con_2, emiss_2])
    return params, app_ref

# Run the model
model_outp = np.zeros((wl, ysize, 64))
parameters = np.zeros((4, ysize, 64))
apparent_refl = np.zeros((wl, ysize, 64))
banddepth = np.zeros((wl, ysize, 64))
ibd = np.zeros((ysize, 64))
emiss_arr = np.zeros((wl, ysize, 64))
# how do i do this
y = 60
x = 40
print(y, x)
parameters[:, y, x], apparent_refl[:, y, x] = model_fitting_15(wave, im, modtran, cos_i, y, x)
banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
print('Ibd: ', ibd[y, x])
y_b = 40
x_b = 30
print(y_b,x_b)
parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b] = model_fitting_15(wave, im, modtran, cos_i, y_b, x_b)
banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
print('Ibd: ', ibd[y_b, x_b])

# plot them both against lab data. 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wavelength_lab, sample1, label='62231')
plt.plot(wavelength_lab, sample2, label='14259')
plt.plot(wavelength_lab, sample3, label='12070')
plt.plot(wavelength_lab, sample4, label='10084')
plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
plt.xlim([1.0, 4.8])
plt.ylim((0, 0.6))
plt.legend()
plt.show() 


