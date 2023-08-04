import numpy as np                                         
import xarray as xr
import math
from scipy.linalg import sqrtm

def letkf_loc_gc(z, L):
    '''
    ================================================================================
    | Gaspari-Cohn localization function.
    | Possibly faster than the Gaussian function, depending on computer architecture.
    | Similar shape to Gaussian, except it is compact, goes to 0 at 2L*sqrt(0.3)
    --------------------------------------------------------------------------------

    Parameters:
        z: value to localize
        L: the equivalent to the Gaussian standard deviation
    
    Notes:
        z and L must have same physical dimension

    Returns:
        localization coeffiecience rloc, rloc~[0,1]
    '''
    c = L / np.sqrt(0.3)
    abs_z = np.abs(z)
    z_c = abs_z / c 

    if abs_z >= 2*c:
        res = 0.0 
    elif abs_z > c:
        res = (0.08333 * z_c**5 -
               0.50000 * z_c**4 +
               0.62500 * z_c**3 +
               1.66667 * z_c**2 -
               5.00000 * z_c +
               4 -
               0.66667 * c/abs_z)
    else:
        res = (-0.25000 * z_c**5 +
               0.50000 * z_c**4 +
               0.62500 * z_c**3 -
               1.66667 * z_c**2 +
               1)
    return res

def get_dist(lat, r):
    """
    Calculate the search distance based on latitude.

    Parameters:
        lat (float): Latitude in degrees.
        r (list or tuple): A list or tuple containing two elements, r[0] and r[1].
    
    Notes: 
        Improvements can be made in a more precise search distance based on lat

    Returns:
        float: The calculated distance.
    """
    dist = ((abs(lat) - 0.0) * r[1] + (90.0 - abs(lat)) * r[0]) / 90.0
    return dist

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth's surface.

    Parameters:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Notes:
        return distance is in [km]

    Returns:
        float: The Haversine distance in kilometers.
    """
    # Earth's radius in kilometers
    earth_radius_km = 6371.0  
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = earth_radius_km * c

    return distance_km

def pletkf(ens_size,x,xlat,xlon,nobs,ylat,ylon,yerr,ymx,hx,loc=True,radius=[500.,50.],infl=1.):
    """
    point by point letkf

    INPUT:
    
      name               dimension             description
      
      ens_size |  scalar: 1               |  ensemble size
      x        |  array : 1 x ens_size    |  background (bkg) members without centralization
      xlat     |  scalar: 1               |  latitude for model grid point in degree 
      xlon     |  scalar: 1               |  longitude for model grid point in degree
      nobs     |  scalar: 1               |  number of observations
      ylat     |  array : nobs x 1        |  latitude for obs in degree
      ylon     |  array : nobs x 1        |  longitude for obs in degree
      yerr     |  array : nobs x 1        |  observational error
      ymx      |  array : nobs x 1        |  y-mean(h(x),axis = ensemble dimension)
      hx       |  matrix: nobs x ens_size |  h(x) for which h is the observation operator, with centralization
      loc      |  bool                    |  OPTIONAL do localization or not, default set to True
      radius   |  array : 2 x 1           |  OPTIONAL localization radius, default set to [500,50] km for eq and polar
      infl     |  scalar: 1               |  OPTIONAL inflation rate to increase bkg spread (>=1), default set to 1

    OUTPUT:

      name               dimension             description
      x        |  array : 1 x ens_size    |  analysis members without centralization
      ns       |  scalar: 1               |  ns <= nobs, valid observations after localization
      olat     |  arrary: ns x 1          |  latitude for obs in degree                                                           [value according to max and min distance] 
      olon     |  arrary: ns x 1          |  longitude for obs in degree                                                          [value according to max and min distance] 
      oerr     |  arrary: ns x 1          |  observational error                                                                  [value according to max and min distance] 
      omx      |  array : ns x 1          |  y-mean(h(x),axis = ensemble dimension)                                               [value according to max and min distance] 
      Hx       |  matrix: ns x ens_size   |  h(x) for which h is the observation operator, with centralization                    [value according to max and min distance]  
      xydist   |  array : ns x 1          |  distance between x and y [km]                                                        [value according to max and min distance]          
      rloc     |  array : ns x 1          |  localization coefficience for each observations (<=1 and >=0), return 1 if loc=false [value according to max and min distance]   
    
    """
    # obs flag where == 1 is finally assimilated, == 0 is omitted
    obs_flag = np.zeros(shape=(nobs))
    # distance between each raw y and x
    distance = np.zeros(shape=(nobs))
    # max search distance
    search   = get_dist(xlat,radius)
    # rloc for each raw y on x
    rd       = np.zeros(shape=(nobs))

    # set loc to false, transfer LETKF back to ETKF
    if(not loc):
        obs_flag[:] = 1
        for s in range(int(nobs)):
            distance[s] = haversine_distance(xlat,xlon,ylat[s],ylon[s])
            rd[s]       = 1.0
    # set loc to true, perform LETKF
    else:
        for s in range(int(nobs)):
            distance[s] = haversine_distance(xlat,xlon,ylat[s],ylon[s])
            rd[s]       = letkf_loc_gc(distance[s],search)
            if(rd[s] > 1e-4):
               obs_flag[s] = 1

    #----------------------------- prepare obseration -------------------------------#
    #  ns         : number of observations which are finally considered in PLETKF
    #  olat, olon : lat and lon for obs
    #  oerr       : R(diagonal matrix, array in practice)
    #  omx        : y - mean(H(x))
    #  Hx         : deviation for each memebers in observation space
    #  xydist     : distance bewteen y and x [km]
    #  rloc       : localization for each y on x

    ns       = int(np.sum  (obs_flag))
    obs_flag = np.where(obs_flag == 1)
    olat     = ylat    [obs_flag]
    olon     = ylon    [obs_flag]
    oerr     = yerr    [obs_flag]
    omx      = ymx     [obs_flag]
    Hx       = hx      [obs_flag,:][0]
    xydist   = distance[obs_flag]
    rloc     = rd      [obs_flag]

    #-------------------------- PLETKF -----------------------------#

    # HxT_Rlinv = Y^{f,T}R^{-1}_{loc}
    HxT_Rlinv = np.zeros(shape=(ens_size,ns))
    for s in range(ns):
        HxT_Rlinv[:,s] = Hx[s,:] / oerr[s] * rloc[s]

    # M1 = Y^{f,T}R^{-1}_{loc}Y^{f}
    M1 = HxT_Rlinv @ Hx

    # M1 = [(N-1)I\infl + Y^{f,T}R^{-1}_{loc}Y^{f}]
    for ens in range(ens_size):
        M1[ens,ens] += float(ens_size-1)/infl

    # Pa = [(N-1)I\infl + Y^{f,T}R^{-1}_{loc}Y^{f}]^{-1}
    Pa = M1.I

    # wa_mean = PaY^{f,T}R^{-1}_{loc}(y^{o}-mean(y^{f}))
    wa_mean = Pa @ HxT_Rlinv @ omx

    # wa_dev = [(N-1)Pa]^{1/2}
    wa_dev  = sqrtm((ens_size-1) * Pa)

    # wa^{i} = wa_mean + wa_dev^{i}, i = 1,2,..,N
    wa = np.zeros(shape=(ens_size,ens_size))

    for ens in range(ens_size):
        wa[:,ens] = wa_dev[:,ens] + wa_mean.T

    # xb_mean
    x_mean = np.mean(x,axis=-1)
    # xb spread
    xb_sprd = np.std(x,ddof=1)
    # xb_dev
    x      = x - x_mean

    # ana_inc = xb_dev * wa
    x = x @ wa
    # xa^{i} = xb_mean + ana_inc^{i}, i = 1,2,..,N
    x = x + x_mean
    xa_sprd = np.std(x,ddof=1)

    idx1 = np.where(xydist == np.max(xydist))
    idx2 = np.where(xydist == np.min(xydist))

    return x,wa_mean.T,wa-np.identity(ens_size),xb_sprd,xa_sprd,ns,search,np.array([xydist[idx1],xydist[idx2]]).T,np.array([rloc[idx1],rloc[idx2]]).T\
        ,np.array([olat[idx1],olat[idx2]]).T,np.array([olon[idx1],olon[idx2]]).T,np.array([oerr[idx1],oerr[idx2]]).T,np.array([omx[idx1],omx[idx2]]).T,Hx