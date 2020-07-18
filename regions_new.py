# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:05:04 2020

@author: klafe
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# this is to try and find, and return a text file, with repeated points over two dates
# define a function which looks for repeated points based on regions in plots
def haversine(lla, y, x, lli):
    R = 1 #1737100
    a = np.sin(np.pi/180 * (lli[1, :, :] - lla[1, y, x])/2)**2 + np.cos(lli[1, :, :]*np.pi/180)* np.cos(lla[1, y, x]*np.pi/180)*np.sin(np.pi/180 * (lli[0, :, :] - lla[0, y, x])/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

def overlap(latlong1, latlong2, ysize):
    '''
    Relies on haversine, size of moon, and assumption of pixel size. 
    latlong1 - the map will be plotted with regards to this one
    latlong2 - regions which appear in latlong2 that have a match in latlong1 will be marked
    ysize - 100 or 135 depending on the cube used. this prevents comparining between day 1/day3 or day2/day3
    
    0.1 was chosen as a limit based off the results from using the moons radius and pixel size. i need a better argument tho
    Returns
    -------
    map_overlap : use as a mask = map == 0 to show region. 

    '''
    mask_o = latlong1[0, :, :] == np.nan
    map_overlap = np.empty((ysize, 64))
    map_overlap[mask_o] = np.nan
    for yj in range(0, ysize):
        for xj in range(0, 64):
            hav = haversine(latlong2, yj, xj, latlong1)
            print(np.shape(hav))
            mask_h = hav <= 0.1
            map_overlap[mask_h] = True
    return map_overlap

def comp_latlong(ll1, ll2):
    test1 = np.isin(ll1[0, :, :], ll2[0, :, :])
    test2 = np.isin(ll1[1, :, :], ll2[1, :, :])
    new2i1 = test1*test2
    plt.imshow(new2i1)
    plt.gca().invert_yaxis()
    plt.title('ll2 in ll1')
    plt.show()    
    
    test1 = np.isin(ll2[0, :, :], ll1[0, :, :])
    test2 = np.isin(ll2[1, :, :], ll1[1, :, :])
    new1i2 = test1*test2
    plt.imshow(new1i2)
    plt.gca().invert_yaxis()
    plt.title('ll1 in ll2')
    plt.show()    
    return new2i1, new1i2

def latlongs(loc, filenames, spatdataname, ysize, sat):
    '''
    Parameters
    ----------
    loc1 : should look like  
    loc = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/fall/research/calibrated_cubes/day352/01/'
    
    filenames : should look like 
    filenames = 'pix_to_longlat_Dec05_1.fit'
    spatdataname : spat_data_day339_scale001.fit or spat_data_day3??_scale030.fit
    
    ysize : will be either 100 or 135, size of the image cube/number of combined slices
    
    Returns
    --------
    positions1, which is where valid data to be considered is in lat long space. 
    '''
    positions1 = np.zeros((2, ysize, 64))
    with fits.open('C:\\Users/klafe/Desktop/spring_senior/research/' + str(filenames)) as hdul:
        positions1[:, :, :] = hdul[0].data
    data = readin(loc+spatdataname, 512, ysize, 64) 
    mask_dark = sat[147, :, :] <= 610 #595 #1.0
    data[147, :, :][mask_dark] = np.nan
    mask_dark1 = np.isnan(data[147, :, :])
    sat_mask = readin(loc + 'sat_mask_regions.fit', 1, ysize, 64)
    sat_mask = np.reshape(sat_mask, (ysize, 64))
    mask = positions1[:, :, :] == -999
    positions1[mask] = np.nan
    positions1[0, :, :][mask_dark1] = np.nan
    positions1[1, :, :][mask_dark1] = np.nan
    mask = positions1[1, :, :] >= -6
    positions1[1, :, :][mask] = np.nan
    positions1[0, :, :] = masking(positions1[0, :, :], sat_mask)
    positions1[1, :, :] = masking(positions1[1, :, :], sat_mask)
    positions1 = np.around(positions1, decimals=0)     
    return positions1

#
# define readin:
def readin(filename, sizez, sizey, sizex):
    # Reads in any fit file into a cube shape
    data = np.zeros((sizez, sizey, sizex))
    with fits.open(filename) as f_hdul:
        data[:, :, :] = f_hdul[0].data
    return data

def readin_sat(loc, ysize):
    # Read in the sat_mask, which is made to exclude specific points which saturate
    sat_mask = readin(loc + 'sat_mask_regions.fit', 1, ysize, 64)
    sat_mask = np.reshape(sat_mask, (ysize, 64))
    return sat_mask

# define ignore regions:
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

#%%
# how can i make these functions more useful

# read in the lat long and im data maps
# File names/locations, changes per cube
loc = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/fall/research/calibrated_cubes/day346/01/'
wave_filename = loc + 'wave_data_day346_scale030.fit'
spec_filename = loc + 'spat_data_day346_scale030.fit'
latlong_loc_1 = 'C:\\Users/klafe/Desktop/spring_senior/research/pix_to_longlat_Dec05_1.fit'
latlong_loc_2 = 'C:\\Users/klafe/Desktop/spring_senior/research/pix_to_longlat_Dec12_1.fit'
sat_mask_filename = loc+'sat_mask_regions.fit'
cosi_filename = 'C:\\Users/klafe/Desktop/spring_senior/research/incidence_ang_Dec12_1.fit'

date_and_scan = np.loadtxt('C:\\Users/klafe/Desktop/spring_senior/research/date_and_scan_bounds.txt', delimiter='\t')
st = 34601
row = np.where(date_and_scan[0, :] == st)
ysize = 100                                 # size of y-array
moon_left = np.int(date_and_scan[1, row])
moon_right = np.int(date_and_scan[2, row])
wl = 512                                    # length of wavelength array

# Read in relevant additions for limiting data and analysis
latlong1 = readin(latlong_loc_1, 2, ysize, 64)
latlong2 = readin(latlong_loc_2, 2, ysize, 64)
sat_mask = readin_sat(loc, ysize)

# Read in the data
im = readin(spec_filename, wl, ysize, 64)
wave = readin(wave_filename, wl, ysize, 64)

# cos_i
cos_ia = np.zeros([ysize, 64])
with fits.open(cosi_filename) as hdul:
    cos_ia[:, :] = hdul[0].data
cos_i = np.cos(np.radians(cos_ia))

# also need n_spec.
n_spec = ignore_regions(latlong2, im, moon_left, moon_right, sat_mask, cos_i)

#%%
# Re write the fuctions to be useful. how do i want to do this?
# what do i want it to look like?
# i think the best way is to just find and match any like i do in the comp_latlong
d1 = readin(latlong_loc_1, 2, ysize, 64)
d2 = readin(latlong_loc_2, 2, ysize, 64)
#out = comp_latlong(d1, d2)

# what is the shpe of hav from overlap?
#test = overlap(d1, d2, ysize)

# maybe just write from scratch:
# how long will it take to do two four loops
i = 0
out = np.zeros((100, 3))
fin = np.zeros((254, 4))#210, 4))
for y in range(0, ysize):
    for x in range(0, 64):
        if np.isnan(n_spec[y, x]) == False:
            hav = haversine(d2, y, x, d1)
            test = np.argwhere(hav <= 0.025)
            if np.size(test[:, 0]) > 1:
                for t in range(0, np.size(test[:, 0])):
                    #print(y, x)
                    # check to see if test values are in n_spec2
                    fin[i, :] = np.array([test[t-1, 0], test[t-1, 1], y, x])
                    # the order, d1 is latlong1 which is test y, x
                    i+=1
                #out[i, :] = np.array([y, x, test])
                #i +=1
print(fin[5, :])
print(latlong1[:, np.int(fin[5, 0]), np.int(fin[5, 1])], latlong2[:, np.int(fin[5, 2]), np.int(fin[5, 3])])#, latlong1[np.int(fin[10, 1])], latlong2[np.int(fin[10, 2])], latlong2[np.int(fin[10, 3])])

mask = np.isnan(n_spec) == False
test= latlong2[0, :, :]
test[mask] = np.nan

plt.scatter(fin[:, 1], fin[:, 0])
plt.imshow(latlong1[0, :, :])
plt.show()

plt.scatter(fin[:, 3], fin[:, 2])
plt.imshow(latlong2[0, :, :])
plt.show()

plt.imshow(test)
plt.show()
# right now this method includes regions in saturated/ off moon/dark regions of the second plug into haversine. 
# lets try switching the order. 
# completely new region. i do not understand 
#%%
old_n_spec = n_spec
# a check: use this across scans instead of days
latlong_loc_2 = 'C:\\Users/klafe/Desktop/spring_senior/research/pix_to_longlat_Dec05_0.fit'
d2 = readin(latlong_loc_2, 2, ysize, 64)

# n_spec depends on d2
loc = 'C:\\Users/klafe/Desktop/prior_to_reset/senior/fall/research/calibrated_cubes/day339/00/'
wave_filename = loc + 'wave_data_day339.fit'
spec_filename = loc + 'spat_data_day339.fit'
sat_mask_filename = loc+'sat_mask_regions.fit'
cosi_filename = 'C:\\Users/klafe/Desktop/spring_senior/research/incidence_ang_Dec05_0.fit'

date_and_scan = np.loadtxt('C:\\Users/klafe/Desktop/spring_senior/research/date_and_scan_bounds.txt', delimiter='\t')
st = 33900
row = np.where(date_and_scan[0, :] == st)
ysize = 100                                 # size of y-array
moon_left = np.int(date_and_scan[1, row])
moon_right = np.int(date_and_scan[2, row])
wl = 512                                    # length of wavelength array

# Read in relevant additions for limiting data and analysis
latlong2 = readin(latlong_loc_2, 2, ysize, 64)
sat_mask = readin_sat(loc, ysize)

# Read in the data
im = readin(spec_filename, wl, ysize, 64)
wave = readin(wave_filename, wl, ysize, 64)

# cos_i
cos_ia = np.zeros([ysize, 64])
with fits.open(cosi_filename) as hdul:
    cos_ia[:, :] = hdul[0].data
cos_i = np.cos(np.radians(cos_ia))

# also need n_spec.
n_spec = ignore_regions(latlong2, im, moon_left, moon_right, sat_mask, cos_i)

i = 0
out = np.zeros((100, 3))
fin = np.zeros((5000, 4))
for y in range(0, ysize):
    for x in range(0, 64):
        if np.isnan(n_spec[y, x]) == False:
            hav = haversine(d2, y, x, d1)
            test = np.argwhere(hav <= 0.025)
            if np.size(test[:, 0]) > 1:
                for t in range(0, np.size(test[:, 0])):
                    #print(y, x)
                    fin[i, :] = np.array([test[t-1, 0], test[t-1, 1], y, x])
                    # the order, d1 is latlong1 which is test y, x
                    i+=1
                #out[i, :] = np.array([y, x, test])
                #i +=1
print(fin[5, :])
print(latlong1[:, np.int(fin[5, 0]), np.int(fin[5, 1])], latlong2[:, np.int(fin[5, 2]), np.int(fin[5, 3])])#, latlong1[np.int(fin[10, 1])], latlong2[np.int(fin[10, 2])], latlong2[np.int(fin[10, 3])])

mask = np.isnan(n_spec) == False
test= latlong2[0, :, :]
test[mask] = np.nan

plt.scatter(fin[:, 1], fin[:, 0])
plt.imshow(latlong1[0, :, :])
plt.show()

plt.scatter(fin[:, 3], fin[:, 2])
plt.imshow(latlong2[0, :, :])
plt.show()

plt.imshow(test)
plt.show()


#%%
# a simple test, 
y = 26
x = 25
havtest = haversine(d2, y, x, d1)
where = np.argwhere(havtest <=0.025)
print(d2[:, y, x], d1[:, 50, 33])

plt.imshow(havtest)
plt.show()

mask = np.isnan(n_spec) == False
t2 = d2[0, :, :]
t2[mask] = 10
plt.imshow(t2)
plt.colorbar()
plt.show()
plt.imshow(d1[0, :, :])
plt.show()
#%%
# something to consider, not excluding the saturated parts of the second one. 


# a check: instead of haversine, just do a basic how different are each value sqrt. 

# returns y, x for both dates, lat long for both
#%%
# trying it outside of a function
# define the inputs and regions to ignore.
import numpy as np

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
    plt.imshow(latlong1[0, :, :])
    plt.show()
    
    plt.scatter(fin[:, 3], fin[:, 2])
    plt.imshow(latlong2[0, :, :])
    plt.show()
    return fin

# test this function
ysize = 100
wl = 512                                    # length of wavelength array       
latlong1 = exclude(34601, ysize)
ysize = 100
latlong2 = exclude(33901, ysize)
results = match_regions(latlong1, latlong2)

d2d1 = results

# now use this to check the values across dates. 
#%%
# save to compare
# we now have d2d1, d2d3
# how do i compare them?
# do np.where match?
# remember, the first column is d2, the second column is the one that changes
#test = np.argwhere(d2d1[:, 0] == d2d3[:, 0])

i = 0 
rep = np.zeros((111))
tst = np.zeros((111, 6))
for a in range(1163):
    if np.any(d2d1[:, 0] == d2d3[a, 0]) and np.any(d2d1[:, 1] == d2d3[a, 1]):
        a = np.int(a)
        rep[i] = a
        where = np.argwhere((d2d1[:, 0] == d2d3[a, 0]) & (d2d1[:, 1] == d2d3[a, 1]))
        if np.size(where) == 1:
            print('one at: ', a)
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
# now, lets check the validity of these points by checking the latlongs for each one
ysize = 135
latlong3 = exclude(35201, ysize)
l = 5

print('day 339: ', latlong2[:, np.int(tst[5, 0]), np.int(tst[5, 1])])
print('day 346: ', latlong1[:, np.int(tst[5, 2]), np.int(tst[5, 3])])
print('day 352: ', latlong3[:, np.int(tst[5, 4]), np.int(tst[5, 5])])
# within 1 degree, so yeah theyre good. 

# lets plot
plt.scatter(tst[:, 1], tst[:, 0])
plt.imshow(latlong2[0, :, :])
# now, with this we can make a new array
#%%

