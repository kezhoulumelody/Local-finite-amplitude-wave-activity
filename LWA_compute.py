```python
import numpy as np
from netCDF4 import Dataset
import glob 
import numpy.ma as ma 

"""
This script is used to calculate local wave activity in various of CMIP single forcing large ensemble 
500 geopotential height for given historical periods
Chen et al (2015): 
"Chen, G., J. Lu, D. A. Burrows, and L. R. Leung (2015), 
Local finite-amplitude wave activity as an objective diagnostic of midlatitude extreme weather
doi:10.1002/2015GL066959."
"""

def find_index(ll, ls):
    """
    Find the index of user defined longitude or latitude
    Arguments:
    ll -- input location matrix lat or lon, must be one dimensional
    ls -- user defined location float or double

    return an integer as index

    """
    index = ma.where(abs(ll - (ls)) == min(abs(ll - (ls))))
    index = int(index[0][0])

    return index

def tracer_eq_1var_2d_local4(lon, lat, lonb, latb, q_tracer, sort_direct='ascend'):
    """ 
    Hybrid Modified Lagrangian-Mean (MLM) and Eulerian diagnostics;
    both Eulerian and Lagrangian variables are output at the input latitudinal grids

    input variables: lon, lat   : grid box centers (degrees)
                     lonb, latb : grid box boundaries (degrees) 
                     q_tracer   : 2d tracer, stored in the format of q_tracer(lon,lat)
                     sort_direct: the direction of sorting q_tracer. default='ascend'; set 'ascend' for PV (values roughly increase with latitude), and 'descend' for Z500 in the NH (values roughly descrease with latitude).
                     NOTE: the direction of lat and latb is assumed to increase with the lat/latb index.

    output variables: qz: Eulerian-mean q
                      Qe: Lagrangian-mean Q
                      Ae: wave activity
                      dQedY: Lagrangian gradient of Q (per radian)
                      AeL: local wave activity as a function of longitude and latitude

    See details in the reference:
    Chen, G., J. Lu, D. A. Burrows, and L. R. Leung, 2015: Local finite-amplitude wave activity as an objective diagnostic of midlatitude extreme weather. Geophys. Res. Lett., 42, 10,952-10,960, doi:10.1002/2015GL066959
    """

    RADIUS = 6.371e6

    lon = np.squeeze(lon)
    lonb = np.squeeze(lonb)
    lat = np.squeeze(lat)
    latb = np.squeeze(latb)

    ni = len(lon)
    nj = len(lat)
    num_point = ni * nj

    sin_latb = np.sin(np.deg2rad(latb))
    cos_lat = np.cos(np.deg2rad(lat))
    sin_lat = np.sin(np.deg2rad(lat))

    # calculate the mass for each grid box
    dX = np.abs(lonb[1:] - lonb[:-1]) * np.pi / 180
    dY = np.abs(sin_latb[1:] - sin_latb[:-1])

    dM2 = np.outer(dX, dY)
    qdM2 = q_tracer * dM2
    # print('calcualte zonal-mean q')
    qz = np.sum(qdM2, axis=0) / np.sum(dM2, axis=0)

    # indices for longitude and latitude
    id = np.tile(np.arange(0, ni), (nj, 1)).T
    jd = np.tile(np.arange(0, nj), (ni, 1))

    # Eulerian integrals between lat(j-1) and lat(j) for j=1,...,nj; The integral between the South Pole and lat(0) or the North Pole and lat(nj) is given half of the mass on a grid.
    dMb = np.zeros(nj+1)
    qdMb = np.zeros(nj+1)
    dMb[nj] = np.sum(dM2[:, nj-1]) * 0.5
    qdMb[nj] = np.sum(qdM2[:, nj-1]) * 0.5
    dMb[0] = np.sum(dM2[:, 0]) * 0.5
    qdMb[0] = np.sum(qdM2[:, 0]) * 0.5
    dMb[1:nj] = np.sum(dM2[:, :nj-1], axis=0) * 0.5 + np.sum(dM2[:, 1:nj], axis=0) * 0.5
    qdMb[1:nj] = np.sum(qdM2[:, :nj-1], axis=0) * 0.5 + np.sum(qdM2[:, 1:nj], axis=0) * 0.5

    # Lagrangian integrals between the two Q contours corresponding to the equivalent latitudes of lat(j-1) and lat(j)
    # sort the tracer field first and then the Lagrangian integral reduces to summation
    # by default, reshape the columns (constant latitude) first
    q1 = np.reshape(q_tracer, num_point, order='F')
    dM1 = np.reshape(dM2, num_point, order='F')
    id1 = np.reshape(id, num_point, order='F')
    jd1 = np.reshape(jd, num_point, order='F')

    if sort_direct == 'ascend':
        q_pos = np.argsort(q1, kind='stable')
    elif sort_direct == 'descend':
        q_pos = np.argsort(q1, kind='stable')[::-1]
    else:
        raise Exception('`sort_direct` should be either `ascend` or `descend`')
    q_sort = q1[q_pos]
    dM1_sort = dM1[q_pos]
    id1_sort = id1[q_pos]
    jd1_sort = jd1[q_pos]

    n = num_point - 1
    Qe = np.zeros(nj)
    dMe = np.zeros(nj+1)
    qdMe = np.zeros(nj+1)
    dMLp = np.zeros((ni, nj))
    AeLp = np.zeros((ni, nj))
    dMLm = np.zeros((ni, nj))
    AeLm = np.zeros((ni, nj))

    # Looping from lat(j=nj-1) to lat(j=0)
    #   For index j, dMb[j] is the Eulerian integral between lat(j-1) and lat(j)
    #   n is the index for the sorted tracer field.
    #   dMe[j] corresponds to the mass between equivalent lat(j-1) and lat(j) or Qe[j-1] and Qe[j]
    #   As n is decreased from num_point-1, dMe[j] is increased until dMe[j] >= dMb[j]
    for j in range(nj, 0, -1):  
        while dMe[j] < dMb[j] and n >= 0:
            dMe[j] += dM1_sort[n]
            qdMe[j] += q_sort[n] * dM1_sort[n]

            # get the indices (i, jQ) for q=q_sort[n]
            i = id1_sort[n]
            jQ = jd1_sort[n]

            #
            # Note for the sorted array, q satisfies that Qe(j-1)<= q <= Qe(j) and we only need to check the lat index for the integral
            #
            # integral for q >= Qe(j-1) and jQ <= j-1; j-1 is the index of equiv lat
            # AeLp denotes that q-Qe is positive (+) or equatorward instrusion
            jj = j-1
            while jQ <= jj:
                if jQ == jj:
                    wgt = 0.5
                else:
                    wgt = 1.0
                dMLp[i, jj] += dM1_sort[n] * wgt
                AeLp[i, jj] += q_sort[n] * dM1_sort[n] * wgt
                jj -= 1
            
            # integral for q <= Qe(j) and jQ >= j; j is the index of equiv lat
            # AeLm denotes that q-Qe is negative (-) or poleward instrusion
            jj = j
            while jQ >= jj:
                if jQ == jj:
                    wgt = 0.5
                else:
                    wgt = 1.0
                dMLm[i, jj] += dM1_sort[n] * wgt
                AeLm[i, jj] += q_sort[n] * dM1_sort[n] * wgt
                jj += 1
            
            # moving on to the next value in q_sort until dMe[j]>=dMb[j]
            n -= 1
        
        # correction for the Q boundary, Qe[j-1], due to nonzero dMe[j]-dMb[j]
        Qe[j-1] = 0.5 * (q_sort[n] + q_sort[n+1])
        dMe[j-1] = dMe[j] - dMb[j]
        qdMe[j-1] = Qe[j-1] * dMe[j-1]
        dMe[j] = dMe[j] - dMe[j-1]
        qdMe[j] = qdMe[j] - qdMe[j-1]
        
        # integral for q >= Q(j-1) and jQ <= j-1; j-1 is the index of equiv lat
        i = id1_sort[n]
        jQ = jd1_sort[n]
        
        jj = j-1
        if jQ <= jj:
            if jQ == jj:
                wgt = 0.5
            else:
                wgt = 1.0
            dMLp[i, jj] -= dMe[j-1] * wgt
            AeLp[i, jj] -= qdMe[j-1] * wgt
        
        # integral for q <= Qe(j-1) and jQ >= j-1; j-1 is the index of equiv lat
        i = id1_sort[n+1]
        jQ = jd1_sort[n+1]

        jj = j-1
        if jQ >= jj:
            if jQ == jj:
                wgt = 0.5
            else:
                wgt = 1.0
            dMLm[i, jj] += dMe[j-1] * wgt
            AeLm[i, jj] += qdMe[j-1] * wgt

    # integral over the rest of q_sort
    dMe[0] += np.sum(dM1_sort[:n+1])
    qdMe[0] += np.sum(q_sort[:n+1] * dM1_sort[:n+1])

    while n >= 0:
        i = id1_sort[n]
        jQ = jd1_sort[n]

        # integral for q <= Qe(0) and jQ >= 0; 0 is the index of equiv lat
        jj = 0
        while jQ >= jj:
            if jQ == jj:
                wgt = 0.5
            else:
                wgt = 1.0
            dMLm[i, jj] += dM1_sort[n] * wgt
            AeLm[i, jj] += q_sort[n] * dM1_sort[n] * wgt
            jj += 1
        n -= 1

    # compute dQedY using finite differencing
    dQedY = np.zeros(nj)
    dQedY[0] = (-1.5 * Qe[0] + 2. * Qe[1] - 0.5 * Qe[2]) / (-1.5 * sin_lat[0] + 2. * sin_lat[1] - 0.5 * sin_lat[2]) * cos_lat[0]
    dQedY[nj-1] = (0.5 * Qe[nj-3] - 2. * Qe[nj-2] + 1.5 * Qe[nj-1]) / (0.5 * sin_lat[nj-3] - 2. * sin_lat[nj-2] + 1.5 * sin_lat[nj-1]) * cos_lat[nj-1]
    dQedY[1:nj-1] = (Qe[:nj-2] - Qe[2:]) / (sin_lat[:nj-2] - sin_lat[2:]) * cos_lat[1:nj-1]

    # Wave activity
    # print('Computing Wave Activity')
    Ae = np.zeros(nj)
    Ae[nj-1] = qdMe[nj] - qdMb[nj]
    for j in range(nj-2, -1, -1):
        Ae[j] = Ae[j+1] + (qdMe[j+1] - qdMb[j+1])
    Ae = Ae * RADIUS / (2 * np.pi) / cos_lat

    # print('Computing local wave activity')
    AeLp = AeLp - np.outer(np.ones(ni), Qe) * dMLp
    AeLp = AeLp * RADIUS / np.outer(dX, cos_lat)

    AeLm = AeLm - np.outer(np.ones(ni), Qe) * dMLm
    AeLm = -AeLm * RADIUS / np.outer(dX, cos_lat)
    # output dy only 
    dYLm = dMLm * RADIUS / np.outer(dX, cos_lat)
    dYLp = dMLp * RADIUS / np.outer(dX, cos_lat)

    AeL = AeLp + AeLm
    # print('AeL Shape is {0}'.format(AeL.shape))

    return qz, Qe, Ae, dQedY, AeL, AeLp, AeLm, dYLm, dYLp, dQedY 

######### Here is an example of calculating LWA-Z500 using codes above ############ 
# read in the lat lon and Z3 daily results  
file = '/<your directory>/<your netCDF file name>'
nc = Dataset(file, 'r')
latt = nc['lat'][:] # rearange the array so that latitude increases with indexs
lon = nc['lon'][:]
la = find_index(latt, 20)
lb = find_index(latt, 90) # only compute Northern Hemisphere (tropics should be exculded)
lat = latt[la:lb+1] 
Z500 = nc['Z500'][:,la:lb+1, :]
time = nc['time'][:]
ntime = len(time)

dlon = 0.5 * abs(lon[1] - lon[0])
lonb = np.zeros([len(lon)+1])
lonb[0] = lon[0] - dlon
lonb[-1] = lon[-1] + dlon
lonb[1:-1] = 0.5 * (lon[0:-1] + lon[1:])

latb = np.zeros([len(latt)+1])
latb[0] = -90 
latb[-1] = 90
latb[1:-1] = 0.5 * (latt[0:-1] + latt[1:])

dirout = '/<your output directory>/'
qz_big = np.zeros([ntime, len(lat)]) # the Eulerian zonal mean of Z500
Qe_big = np.zeros(qz_big.shape) # The largrangian zonal mean of Z500
Ae_big = np.zeros(qz_big.shape) # The wave activity
dQdy_big = np.zeros(qz_big.shape) # Meridional gradient of the Largrangian zonal mean of Z500
AeLp_big = np.zeros([ntime, len(lat), len(lon)]) # The local wave activity above Qe contour
AeLm_big = np.zeros([ntime, len(lat), len(lon)]) # The local wave activity below Qe contour
dYLm_big = np.zeros([ntime, len(lat), len(lon)])
dYLp_big = np.zeros([ntime, len(lat), len(lon)]) 
 
# Calculate LWA for each time step using for loop
for nk in range(ntime):
    tq = Z500[nk, :, :]
    theta = tq.T

    # calculate the wave activity
    qz, Qe, Ae, dQedY, AeL, AeLp, AeLm, dYLm, dYLp, dQdy = tracer_eq_1var_2d_local4(lon, lat, lonb, latb[la:lb+2], theta, sort_direct='descend')
    qz_big[nk, :] = qz
    Qe_big[nk, :] = Qe
    Ae_big[nk, :] = Ae 
    dQdy_big[nk, :] = dQdy
    AeLp_big[nk, :, :] = AeLp.T
    AeLm_big[nk, :, :] = AeLm.T
    dYLm_big[nk, :, :] = dYLm.T 
    dYLp_big[nk, :, :] = dYLp.T

# Save the output as netCDF file 
casename = dirout+'<your output file name>'
ncfile = Dataset(casename, mode='w', format='NETCDF4')
lat_dim = ncfile.createDimension('lat', len(lat))     # latitude axis
lon_dim = ncfile.createDimension('lon', len(lon))    # longitude axis
time_dim = ncfile.createDimension('time', len(time))

nlat = ncfile.createVariable('lat', 'f8', ('lat',))
nlat.units = 'degrees_north'
nlat.long_name = 'latitude'
nlat.standard_name = 'latitude'
nlat.axis = 'X'

nlon = ncfile.createVariable('lon', 'f8', ('lon',))
nlon.units = 'degrees_east'
nlon.long_name = 'longitude'
nlon.standard_name = 'longitude'
nlon.axis = 'Y'

nT = ncfile.createVariable('time', 'f8', ('time',))
nT.units = 'days since 1950-01-01 00:00:00'
nT.long_name = 'time'
nT.standard_name = 'time'
nT.calendar = 'noleap'
nT.axis='T'

nqz = ncfile.createVariable('qz', 'f8', ('time', 'lat')) # time dimension sits in the leftmost
nqz.units = 'm'
nqz.standard_name = 'Eulerian-mean Z500'

nqe = ncfile.createVariable('Qe', 'f8', ('time', 'lat')) # time dimension sits in the leftmost
nqe.units = 'm'
nqe.standard_name = 'Lagrangian-mean Z500'

nqy = ncfile.createVariable('dQedy', 'f8', ('time', 'lat')) # time dimension sits in the leftmost
nqy.units = 'm / m'
nqy.standard_name = 'Meridional gradient of Lagrangian-mean Z500'

nae = ncfile.createVariable('Ae', 'f8', ('time', 'lat')) # time dimension sits in the leftmost
nae.units = 'm*m'
nae.standard_name = 'Wave activity of Z500'

nalp = ncfile.createVariable('AeLp', 'f8', ('time', 'lat', 'lon')) # time dimension sits in the leftmost
nalp.units = 'm*m'
nalp.standard_name = 'Local Wave activity of Z500 - up'

nalm = ncfile.createVariable('AeLm', 'f8', ('time', 'lat', 'lon')) # time dimension sits in the leftmost
nalm.units = 'm*m'
nalm.standard_name = 'Local Wave activity of Z500 - below'

nyp = ncfile.createVariable('dYLp', 'f8', ('time', 'lat', 'lon')) # time dimension sits in the leftmost
nyp.units = 'm'
nyp.standard_name = 'Latitudinal displacement around equivalent latitude - up'

nym = ncfile.createVariable('dYLm', 'f8', ('time', 'lat', 'lon')) # time dimension sits in the leftmost
nym.units = 'm'
nym.standard_name = 'Latitudinal displacement around equivalent latitude - below'

nlat[:] = lat
nlon[:] = lon
nT[:] = time
nqz[:] = qz_big
nqe[:] = Qe_big
nae[:] = Ae_big
nalp[:] = AeLp_big
nalm[:] = AeLm_big
nyp[:] = dYLp_big
nym[:] = dYLm_big
nqy[:] = dQdy_big

ncfile.close()







