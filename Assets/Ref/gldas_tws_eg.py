import os
from netCDF4 import Dataset
import numpy as np

#===========================================================================

# import time
# from datetime import datetime, timedelta
# from netCDF4 import num2date, date2num
# nc_file = 'F:/My_Postdoctor/GLDAS/GLDAS_NOAH025_M.A200001.021.nc4.SUB.nc4'

month2seconds = 30*24*3600 # seconds in a month
month2threehours = 30*8 # numbers of 3 hours in a month
filelist = []  # create a file list for GLDAS data
TWSCs = []     # create TWSCs (changes of Terrestrial Water Storage)
PRCPs = []
ETs = []
Qss = []
Qsbs = []
# read GLDAS files of *.nc4 (0.25 x 0.25)，and put them into filelist
for (dirname, dirs, files) in os.walk('F:/My_Postdoctor/GlobalGLDAS/'):
    for filename in files:
        if filename.endswith('.nc4'):
            filelist.append(os.path.join(dirname,filename))

filelist = np.sort(filelist) # order in time
num_files = len(filelist)    # 201 files of *.nc4 from 2002.4-2018.12

# read each file of *.nc4
n=1000
for files in filelist:
    data = Dataset(files, 'r',format='netCDF4')

    lons_gldas = data.variables['lon'][:]          #len: 190
    lats_gldas = data.variables['lat'][:]          #len: 103

    precipitation_flux = data.variables['Rainf_f_tavg'][:]
    water_evaporation_flux = data.variables['Evap_tavg'][:]
    surface_runoff_amount = data.variables['Qs_acc'][:]
    subsurface_runoff_amount = data.variables['Qsb_acc'][:]

    data.close() # close files of GLDAS

    # get monthly P, ET, Surface runoff, underground runoff
    precipitation_flux = precipitation_flux[0]          # from (1,600,1440) to （600,1440) to reduce dim
    water_evaporation_flux = water_evaporation_flux[0]
    surface_runoff_amount = surface_runoff_amount[0]
    subsurface_runoff_amount = subsurface_runoff_amount[0]

    # calculate change of (k-1)-month TWSC in term of k-month (k>=1) TWSC
    TWSC = (precipitation_flux - water_evaporation_flux) * month2seconds - (surface_runoff_amount + subsurface_runoff_amount) * month2threehours
    PRCP = precipitation_flux * month2seconds
    ET   = water_evaporation_flux * month2seconds
    Qs   = surface_runoff_amount * month2threehours
    Qsb  = subsurface_runoff_amount * month2threehours

    # #save
    # lons_new = lons_gldas[900:1072]
    # lats_new = lats_gldas[372:468]
    # PRCP = precipitation_flux[372:468, 900:1072]*month2seconds
    # ET = water_evaporation_flux[372:468, 900:1072]*month2seconds
    # Qs = surface_runoff_amount[372:468, 900:1072]* month2threehours
    # Qsb = subsurface_runoff_amount[372:468, 900:1072]* month2threehours
    # TWSC = TWSC[372:468, 900:1072]
    # n+= 1
    # np.savetxt('F:/My_Postdoctor/TWS_project/TWS_GLDAS/GLDAS_TWSC/PRCP_' + str(n), PRCP)
    # np.savetxt('F:/My_Postdoctor/TWS_project/TWS_GLDAS/GLDAS_TWSC/ET_' + str(n), ET)
    # np.savetxt('F:/My_Postdoctor/TWS_project/TWS_GLDAS/GLDAS_TWSC/Qs_' + str(n), Qs )
    # np.savetxt('F:/My_Postdoctor/TWS_project/TWS_GLDAS/GLDAS_TWSC/Qsb_' + str(n), Qsb)
    # np.savetxt('F:/My_Postdoctor/TWS_project/TWS_GLDAS/GLDAS_TWSC/lon_'+ str(n), lons_new)
    # np.savetxt('F:/My_Postdoctor/TWS_project/TWS_GLDAS/GLDAS_TWSC/lat_'+ str(n), lats_new)


    # GLDAS 月文件中的海洋区域和南极洲地区均无有效测量值(默认填充为 −9999.0)，这里将填充值重设为零。
    TWSCs.append(TWSC.filled(0))


TWSCs = np.array(TWSCs)
# 首先计算第 k(k>=1) 个月相对于第 0 个月的陆地水储量变化，即每个月的陆地水储量变化，然后计算陆地水储量变化的月平均
TWSCs_acc = np.cumsum(TWSCs, axis=0)
TWSCs_acc_average = np.average(TWSCs_acc, axis=0)
# 对每个月的陆地是储量变化进行去平均化，and get the final TWSCs
TWSCs = TWSCs_acc - TWSCs_acc_average

#save ewt
# n=1000
# for da in TWSCs:
#     n+=1
#     np.savetxt('F:/My_Postdoctor/TWS_project/TWS_GLDAS/GLDAS_TWSC/TWSC_'+str(n),da)


