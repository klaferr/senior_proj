# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:26:10 2020

@author: klafe

How to run: input the dates you want to compare, then pick a y, x thats repeated across them and compare output values. 
can i put the second part into a loop?
"""
#comparing across
# general
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy import constants as const
from scipy.optimize import curve_fit
import math

# Define constants
h = const.h
c = const.c
k = const.k

wl = 512

# which dates/scans:
st1 = 34601
st2 = 33901  
st3 = 35201   

def readin(filename, sizez, sizey, sizex):
    # Reads in any fit file into a cube shape
    data = np.zeros((sizez, sizey, sizex))
    with fits.open(filename) as f_hdul:
        data[:, :, :] = f_hdul[0].data
    return data

def resample(m_wave, m_rad, wave_data):
    # Re-sampling modtran data to fit the length of the provided data
    f = interp1d(m_wave, m_rad, fill_value="extrapolate")
    resampled_wave = f(wave_data)
    return resampled_wave


def ignore_regions(latlong, ysize, moon_left, moon_right, sat, incidence):
    lat = latlong[1, :, :]
    mask = lat > -5
    n_spec = np.ones((ysize, 64), dtype=bool)
    n_spec[mask] = False #np.nan    
    long = latlong[0, :, :]
    mask = long < 0
    n_spec[mask] = False #np.nan
    t = np.zeros((ysize, 64))
    t[:, 0:moon_left] = 1
    t[:, moon_right:64] = 1
    mask = t == 1
    n_spec[mask] = False #np.nan  
    mask = sat == 0
    n_spec[mask] = False #np.nan
    regions = n_spec
    mask = incidence < 0
    regions[mask] = False #np.nan
    
    return regions


def readin_sat(loc, ysize):
    # Read in the sat_mask, which is made to exclude specific points which saturate
    sat_mask = readin(loc + 'sat_mask_regions.fit', 1, ysize, 64)
    sat_mask = np.reshape(sat_mask, (ysize, 64))
    return sat_mask

temp_guess = 360
m_guess = 0.01
const_guess =  0.02
e_guess = 0.9

#%%
# 1. find repeats in dates/scans
def haversine(lla, y, x, lli):
    R = 1 #1737100
    a = np.sin(np.pi/180 * (lli[1, :, :] - lla[1, y, x])/2)**2 + np.cos(lli[1, :, :]*np.pi/180)* np.cos(lla[1, y, x]*np.pi/180)*np.sin(np.pi/180 * (lli[0, :, :] - lla[0, y, x])/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

def exclude(st, ysize):
    if st == 33901:
        loc = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/fall/research/calibrated_cubes/day339/01/'
        latlong_loc = 'C:\\Users/klafe/Desktop/spring_senior/research/pix_to_longlat_Dec05_1.fit'
        cosi_filename = 'C:\\Users/klafe/Desktop/spring_senior/research/incidence_ang_Dec05_1.fit'
    elif st == 34601:
        loc = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/fall/research/calibrated_cubes/day346/01/'
        latlong_loc = 'C:\\Users/klafe/Desktop/spring_senior/research/pix_to_longlat_Dec12_1.fit'
        cosi_filename = 'C:\\Users/klafe/Desktop/spring_senior/research/incidence_ang_Dec12_1.fit'
    elif st == 35201:
        loc = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/fall/research/calibrated_cubes/day352/01/'
        latlong_loc = 'C:\\Users/klafe/Desktop/spring_senior/research/pix_to_longlat_Dec18_1.fit'
        cosi_filename = 'C:\\Users/klafe/Desktop/spring_senior/research/incidence_ang_Dec18_1.fit'
    
    ll = readin(latlong_loc, 2, ysize, 64)
    sat = readin_sat(loc, ysize)
    
    # cos_i
    cos_ia = np.zeros([ysize, 64])
    with fits.open(cosi_filename) as hdul:
        cos_ia[:, :] = hdul[0].data
    inc = np.cos(np.radians(cos_ia))
    
    date_and_scan = np.loadtxt('C:\\Users/klafe/Desktop/spring_senior/research/date_and_scan_bounds.txt', delimiter='\t')
    row = np.where(date_and_scan[0, :] == st)                         # size of y-array
    ml = np.int(date_and_scan[1, row])
    mr = np.int(date_and_scan[2, row])
    
    spec = ignore_regions(ll, ysize, ml, mr, sat, inc)
    mask = spec == False
    ll[:, mask] = np.nan
    
    return ll 

# write this into a functions
def match_regions(latlong1, latlong2):
    # now loop
    fin = np.ones((1170, 4))*np.nan
    i = 0
    for y in range(0, ysize):
        for x in range(0, 64):
            #if np.isnan(n_spec_1[y, x]) == False:
            hav = haversine(latlong2, y, x, latlong1)
            test = np.argwhere(hav <= 0.015)
            if np.size(test[:, 0]) > 0:
                for t in range(0, np.size(test[:, 0])):
                #print(y, x)
                    fin[i, :] = np.array([test[t-1, 0], test[t-1, 1], y, x])
                    # the order, d1 is latlong1 which is test y, x
                    i+=1
    print(fin[5, :])
    print(latlong1[:, np.int(fin[5, 0]), np.int(fin[5, 1])], latlong2[:, np.int(fin[5, 2]), np.int(fin[5, 3])])#, latlong1[np.int(fin[10, 1])], latlong2[np.int(fin[10, 2])], latlong2[np.int(fin[10, 3])])
    
    
    plt.scatter(fin[:, 1], fin[:, 0])
    ar = np.array([0, 6, 14, 29, 43, 59, 74, 90, 103, 119, 132, 146, 161, 193, 212])
    for i in ar:
        plt.scatter(fin[i, 1], fin[i, 0], color='r')
    plt.imshow(latlong1[0, :, :])
    plt.gca().invert_yaxis()
    plt.title('346 01')
    plt.show()
    
    plt.scatter(fin[:, 3], fin[:, 2])
    for i in ar:
        plt.scatter(fin[i, 3], fin[i, 2], color='r')
    plt.imshow(latlong2[0, :, :])
    plt.gca().invert_yaxis()
    plt.title('339 01')
    plt.show()
    return fin

# test this function
wl = 512
ysize = 100       
latlong1 = exclude(st1, ysize)
im_346_01 = readin('C:\\Users/klafe/Desktop/prior_to_reset/senior/fall/research/calibrated_cubes/day346/01/spat_data_day346_scale030.fit', wl, ysize, 64)

plt.imshow(im_346_01[147, :, :])
plt.imshow(latlong1[0, :, :], alpha=0.5)

plt.show()
#%%
ysize = 100
latlong2 = exclude(st2, ysize)
results = match_regions(latlong1, latlong2)
d2d1 = results

ysize = 100             
latlong1 = exclude(st1, ysize)
ysize = 135
latlong3 = exclude(st3, ysize)
results = match_regions(latlong1, latlong3)
d2d3 = results
#%%
i = 0 
rep = np.zeros((111))
tst = np.zeros((111, 6))*np.nan
for a in range(1163):
    if np.any(d2d1[:, 0] == d2d3[a, 0]) and np.any(d2d1[:, 1] == d2d3[a, 1]):
        a = np.int(a)
        rep[i] = a
        where = np.argwhere((d2d1[:, 0] == d2d3[a, 0]) & (d2d1[:, 1] == d2d3[a, 1]))
        if np.size(where) == 1:
            where = np.int(where)
            tst[i, :] = np.array([d2d1[where, 2], d2d1[where, 3], d2d1[where, 0], d2d1[where, 1], d2d3[a, 2], d2d3[a, 3]])
            i+=1
        elif np.size(where) > 1:
            print('multiple at: ', a)
            n = np.size(where[0])
            for k in n:
                wheren = np.int(where[n])
                tst[i, :] = np.array([d2d1[where, 2], d2d1[where, 3], d2d1[where, 0], d2d1[where, 1], d2d3[a, 2], d2d3[a, 3]])
                i+=1
                
#%%
plt.scatter(tst[:, 1], tst[:, 0], color='red')
plt.scatter(tst[50, 1], tst[50, 0], color='black')
plt.imshow(latlong2[0, :, :])
plt.show()

plt.scatter(tst[:, 3], tst[:, 2], color='red')
plt.scatter(tst[50, 3], tst[50, 2], color='black')

plt.imshow(latlong1[0, :, :])
plt.show()

plt.scatter(tst[:, 5], tst[:, 4], color='red')
plt.scatter(tst[50, 5], tst[50, 4], color='black')

plt.imshow(latlong3[0, :, :])
plt.show()



#%%
# 2. run M14

#14. emissivity fit as a wavelength dependent value
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

def scatter(wavelength, m, constant):
    # Scattered part
    return m * wavelength + constant

def planck(wavelength, temperature, emiss):
    # Thermal part
    wavelength = wavelength * 10 ** (-6)
    bot = math.e ** (h * c / (wavelength * k * temperature)) - 1
    top = (2 * h * c ** 2) / ((wavelength ** 5) * bot)
    top = top*10**(-6)
    them = (1-emiss)*top 
    return them

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
    #print(opt[0], opt[1])
    
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
    m_3 = opt[0]
    con_3 = opt[1]
    
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
    print('temps throught steps: ',temp_1, temp_2, temp_4)
    print('m through steps: ', m_2,  m_3)
    print('c through: ', con_2, con_3)
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
    plt.plot(wave[:, y, x], thermal_part_2*np.pi/(modtran[:, y, x]*cosi)+scat_part_2, label='fit 1')
    plt.plot(wave[:, y, x], thermal_part_3*np.pi/(modtran[:, y, x]*cosi)+new_scat, label='fit 2')

    
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
    
    params = np.array([temp_4, m_3, con_3, np.nanmean(emiss_2)])
    return params, app_ref, emiss_2

wavelength_lab, sample1, sample2, sample3, sample4 = np.loadtxt('C:\\Users/klafe\Desktop/spring_senior/research/Lunar_Soils_Lab.txt', skiprows=1, unpack=True)
wavelength_lab = wavelength_lab/1000

# ************************************Defined based on the date. 
def m14(st):
    if st == 33901:
            loc = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/fall/research/calibrated_cubes/day339/01/'
            cosi_filename = 'C:\\Users/klafe/Desktop/spring_senior/research/incidence_ang_Dec05_1.fit'
            latlong_loc = 'C:\\Users/klafe/Desktop/spring_senior/research/pix_to_longlat_Dec05_1.fit'
            ysize = 100
            y = 61#np.int(d2d1[5, 2])
            x = 14#np.int(d2d1[5, 3])
            wave = readin(loc+'wave_data_day339_scale001.fit', wl, ysize, 64)
            im = readin(loc+'spat_data_day339_scale001.fit', wl, ysize, 64)
    elif st == 34601:
            loc = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/fall/research/calibrated_cubes/day346/01/'
            cosi_filename = 'C:\\Users/klafe/Desktop/spring_senior/research/incidence_ang_Dec12_1.fit'
            latlong_loc = 'C:\\Users/klafe/Desktop/spring_senior/research/pix_to_longlat_Dec12_1.fit'
            ysize = 100
            y = 59#np.int(d2d1[5, 0])
            x = 37#np.int(d2d1[5, 1])
            wave = readin(loc+'wave_data_day346_scale030.fit', wl, ysize, 64)
            im = readin(loc+'spat_data_day346_scale030.fit', wl, ysize, 64)
    elif st == 35201:
            loc = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/fall/research/calibrated_cubes/day352/01/'
            cosi_filename = 'C:\\Users/klafe/Desktop/spring_senior/research/incidence_ang_Dec18_1.fit'
            latlong_loc = 'C:\\Users/klafe/Desktop/spring_senior/research/pix_to_longlat_Dec18_1.fit'
            ysize = 135
            y = 54#np.int(d2d3[5, 2])
            x = 12#np.int(d2d3[5, 3])
            wave = readin(loc+'wave_data_day352_scale030.fit', wl, ysize, 64)
            im = readin(loc+'spat_data_day352_scale030.fit', wl, ysize, 64)
    
     
    sat = readin_sat(loc, ysize)
    
    # cos_i
    cos_ia = np.zeros([ysize, 64])
    with fits.open(cosi_filename) as hdul:
        cos_ia[:, :] = hdul[0].data
    cos_i = np.cos(np.radians(cos_ia))
    
    #sat regions
    date_and_scan = np.loadtxt('C:\\Users/klafe/Desktop/spring_senior/research/date_and_scan_bounds.txt', delimiter='\t')
    row = np.where(date_and_scan[0, :] == st)                         # size of y-array
    ml = np.int(date_and_scan[1, row])
    mr = np.int(date_and_scan[2, row])
    
    ll = readin(latlong_loc, 2, ysize, 64)
    
    n_spec = ignore_regions(ll, ysize, ml, mr, sat, cos_i)
    
    
    modtran_filename = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/spring/research/Modtran_1nm_4.85um.txt'
    modtran_wave, modtran_rad = np.loadtxt(modtran_filename, skiprows=1, unpack=True)
    modtran = resample(modtran_wave, modtran_rad, wave)
    mod = modtran
    
    
    # Run the model
    model_outp = np.zeros((wl, ysize, 64))
    parameters = np.zeros((4, ysize, 64))
    apparent_refl = np.zeros((wl, ysize, 64))
    banddepth = np.zeros((wl, ysize, 64))
    ibd = np.zeros((ysize, 64))
    emiss_arr = np.zeros((wl, ysize, 64))
    
    print(y, x)
    parameters[:, y, x], apparent_refl[:, y, x], emiss_arr[:, y, x] = model_fitting_14(wave, im, modtran, cos_i, y, x)
    banddepth[:, y, x], ibd[y, x] = finding_ibanddepth(wave, apparent_refl, y, x, '3')
    print('Ibd: ', ibd[y, x])
    
    #y_b = 40
    #x_b = 30
    #print(y_b,x_b)
    #parameters[:, y_b, x_b], apparent_refl[:, y_b, x_b], emiss_arr[:, y_b, x_b] = model_fitting_14(wave, im, modtran, cos_i, y_b, x_b)
    #banddepth[:, y_b, x_b], ibd[y_b, x_b] = finding_ibanddepth(wave, apparent_refl, y_b, x_b, '3')
    #print('Ibd: ', ibd[y_b, x_b])
    
    # plot them both against lab data. 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(wavelength_lab, sample1, label='62231')
    plt.plot(wavelength_lab, sample2, label='14259')
    plt.plot(wavelength_lab, sample3, label='12070')
    plt.plot(wavelength_lab, sample4, label='10084')
    plt.plot(wave[:, y, x], apparent_refl[:, y, x], label='(%2.0f' %y+','+'%2.0f)'%x)
    #plt.plot(wave[:, y_b, x_b], apparent_refl[:, y_b, x_b], label='(%2.0f' %y_b+','+'%2.0f)'%x_b)
    plt.xlim([1.0, 4.8])
    plt.ylim((0, 0.6))
    plt.legend()
    plt.show() 
    
    
    print(parameters[:, y, x])
    print(ibd[y, x])
    return parameters[:, y, x]

print(st1)
d34601 = m14(st1)
print(st2)
d33901 = m14(st2)
print(st3)
d35201 = m14(st3)

# 3. compare output values