# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:40:16 2021
@author: thomas.berthou
The Most Simple Building Energy Model is for students and teachers to learn and teach about energy building simulation through a simple exemple
This model simulate heating and cooling needs for a buildings at a chosen time step (maximum is 1800 seconds)
What to do with this model :
 - Load shedding assessment
 - Sensitivity analysis
 - Refurbishing assessment
 - Calibration from on site measurement or detail simulation
 - Model Predictive Control
 - Building stock simulation
"""
import numpy as np
import matplotlib.pyplot as plt

#%% Weather data and other solicitations at 1 hour time step for one year
#TODO : read external weather files for more realistic simulation

#Outdoor temperature
t_out = 25*np.sin(np.linspace(0, np.pi, 8760)) + np.tile(5*np.sin(np.linspace(-np.pi, np.pi, 24)), 365)  

#Heating setpoint temperature
t_set_winter = 21*np.ones(24)
t_set_winter[0:7] += -4
t_set_winter = np.tile(t_set_winter,365)
t_set_winter[120*24:300*24] = 12 #no heating in summer

#Cooling setpoint temperature
t_set_summer = 25*np.ones(24)
t_set_summer = np.tile(t_set_summer,365)
t_set_summer[0:120*24] = 30  #no cooling in winter
t_set_summer[300*24:] = 30 #no cooling in winter

#Internal heat load from occupancy (appliances and humans)
internal_load_occ = 250 * np.ones(24)
internal_load_occ[[7,8,9,12,13,18,19,20,21,22]] = 2500
internal_load_occ = np.tile(internal_load_occ,365)

#Internal heat load from solar flux passing through windows TODO : more realistic solar flux
internal_load_solar = 0*np.ones(24)
internal_load_solar[8:19] = 100*np.array([1,2,3,4,5,6,5,4,3,2,1])
internal_load_solar = np.tile(internal_load_solar,365) * 5*np.sin(np.linspace(0, np.pi, 8760))
#External heat load from solar flux which is absorbed by walls TODO : more realistic solar flux
external_load_solar = 5*internal_load_solar

rc_solicitation = dict()
rc_solicitation['t_out'] = t_out
rc_solicitation['t_set_winter'] = t_set_winter
rc_solicitation['t_set_summer'] = t_set_summer
rc_solicitation['internal_load_occ'] = internal_load_occ
rc_solicitation['internal_load_solar'] = internal_load_solar
rc_solicitation['external_load_solar'] = external_load_solar

#%% Building characteristics for RC model parameters calculation
rho_air = 1.2 # air density (kg.m-3)
c_air = 1004 # air heat capacity (J.K^-1.kg^-1)
s_floor = 100 # living floor [m²]
s_in = 435 # opaque walls surface (vertical and horizontal) in contact with indoor temperature [m²]
s_out = 235 # opaque walls surface (vertical and horizontal) in contact with outdoor temperature [m²]
s_windows = 15 # surface of windows [m²]
u_out = 1.5 # mean U value of opaques walls (vertical and horizontal) [W/(K.m²)]
u_windows = 2.5 #mean U value of windows (conduction and convection included) [W/(K.m²)]
h_in = 6 # indoor convection coefficient [W/(K.m²)]
h_out = 20 # outdoor convection coefficient [W/(K.m²)]
m_air_new = 0.6 # mass flow rate of fresh air [Vol/hour]
v_in = 250 # indoor air volume [m3]
inertia_surf = 432000 # surface inertia of building structure [J/K.m²]
rc_parameters = dict() # dictionary with the R and C values
rc_parameters['r_conv_ext'] = 1/(h_out*s_out)
rc_parameters['r_cond_wall'] = 1/(u_out*s_out)
rc_parameters['r_conv_int'] = 1/(h_in*s_in)
rc_parameters['r_infiltration'] = 1/(rho_air*c_air*m_air_new*v_in/3600)
rc_parameters['r_wondows'] = 1/(u_windows*s_windows)
rc_parameters['C_air'] = v_in * c_air * rho_air * 15 # add inertia of furniture and light walls (x15)
rc_parameters['C_wall'] = inertia_surf * s_floor
    
#%% Simulation parameters
delta = 600
simu_parameters = dict()
simu_parameters['delta'] = delta # simulation time step in second (300 seconds to 1800 seconds)
simu_parameters['p_heat_max'] = [100 * s_floor]* len(t_out) # maximum heat delivered 100 W per m² (watt)
simu_parameters['p_cold_max'] = [-50 * s_floor] * len(t_out) # maximum cold delivered  50 W per m² (watt)
simu_parameters['start'] = 0 # first day of simulation
simu_parameters['stop'] = 365 # last day of simulation
#TODO simulate a large commercial building in place of a dwelling

def R6C2 (simu_parameters, rc_solicitation, rc_parameters) :
    '''
    adapted from Berthou et al. 2014 : Development and validation of a gray box model 
    to predict thermal behavior of occupied office buildings, Energy and Buildings
    TODO : more stable discretization scheme, ground temperature, add HVAC systems... 
    '''
    #Simulation parameters unpack
    delta = simu_parameters['delta']
    p_heat_max = simu_parameters['p_heat_max']
    p_heat_max = np.repeat(p_heat_max, int(3600/delta))
    p_cold_max = simu_parameters['p_cold_max']
    p_cold_max = np.repeat(p_cold_max, int(3600/delta))
    start = int(simu_parameters['start'] * 24*3600/delta)
    stop = int(simu_parameters['stop'] * 24*3600/delta)
    
    #RC values from rc_parameters
    rg = rc_parameters['r_wondows']
    rv = rc_parameters['r_infiltration']
    re = rc_parameters['r_conv_ext']
    rw = rc_parameters['r_cond_wall']
    rs = rc_parameters['r_conv_int']
    ri = rc_parameters['r_conv_int']
    ci = rc_parameters['C_air']
    cw = rc_parameters['C_wall']
    
    #Solicitation from rc_solicitation adapted to chosen simulation time step
    alpha = 0.5 # radiative part of internal loads
    source1 = np.repeat((1-alpha) * (rc_solicitation['internal_load_occ'] + rc_solicitation['internal_load_solar']), int(3600/delta))
    source2 = np.repeat(alpha * (rc_solicitation['internal_load_occ'] + rc_solicitation['internal_load_solar']), int(3600/delta))
    source3 = np.repeat(rc_solicitation['external_load_solar'], int(3600/delta))
    t_out = np.repeat(rc_solicitation['t_out'], int(3600/delta))
    t_set_winter = np.repeat(rc_solicitation['t_set_winter'], int(3600/delta))
    t_set_summer = np.repeat(rc_solicitation['t_set_summer'], int(3600/delta))
    
    #Initial values of ti (indoor temperature), tw (walls temperature) and powers
    ti = t_set_winter[start]
    tw = t_set_winter[start]
    p_heat = 0
    p_cold = 0
       
    #Lists to store values of interest 
    heating_need = [0] #watt
    cooling_need = [0] #watt
    t_in = [ti] # indoor temperature [°C]
    t_wall = [tw] # wall temperature [°C]
    for i in range(start+1,stop): #loop over time (one year), Euler explicit for resolution (stable under condition !)
        ts = (ti/ri + tw/rs + source2[i])/(1/ri + 1/rs)
        th = (tw/rw + t_out[i]/re + source3[i])/(1/rw + 1/re)
        tw = ((th-tw)/rw + (ts-tw)/rs)*delta/cw + tw
        ti = ((ts-ti)/ri + (t_out[i]-ti)/rg + (t_out[i]-ti)/rv + source1[i] + p_heat + p_cold)*delta/ci + ti
         
        p_heat = ci*(t_set_winter[i]-ti)/delta + (ti-ts)/ri + (ti-t_out[i])/rg + (ti-t_out[i])/rv - source1[i]
        p_heat  = np.min([np.max([0,p_heat]),p_heat_max[i]])
        p_cold = ci*(t_set_summer[i]-ti)/delta + (ti-ts)/ri + (ti-t_out[i])/rg + (ti-t_out[i])/rv - source1[i]
        p_cold = np.max([np.min([0,p_cold]),p_cold_max[i]])
        
        heating_need.append(p_heat)
        cooling_need.append(p_cold)
        t_in.append(ti)
        t_wall.append(tw)
    
    return (np.array(heating_need), np.array(cooling_need), np.array(t_in), np.array(t_wall))

#%% R6C2 dynamic thermal model (no HVAC system)
def R6C2_crank_nicolson(simu_parameters, rc_solicitation, rc_parameters) :
    '''
    adapted from Berthou et al. 2014 : Development and validation of a gray box model 
    to predict thermal behavior of occupied office buildings, Energy and Buildings
    TODO : more stable discretization scheme, ground temperature, add HVAC systems... 
    '''
    #Simulation parameters unpack
    delta = simu_parameters['delta']
    p_heat_max = simu_parameters['p_heat_max']
    p_heat_max = np.repeat(p_heat_max, int(3600/delta))
    p_cold_max = simu_parameters['p_cold_max']
    p_cold_max = np.repeat(p_cold_max, int(3600/delta))
    start = int(simu_parameters['start'] * 24*3600/delta)
    stop = int(simu_parameters['stop'] * 24*3600/delta)
    
    #RC values from rc_parameters
    rg = rc_parameters['r_wondows']
    rv = rc_parameters['r_infiltration']
    re = rc_parameters['r_conv_ext']
    rw = rc_parameters['r_cond_wall']
    rs = rc_parameters['r_conv_int']
    ri = rc_parameters['r_conv_int']
    ci = rc_parameters['C_air']
    cw = rc_parameters['C_wall']
    
    #Solicitation from rc_solicitation adapted to chosen simulation time step
    alpha = 0.5 # radiative part of internal loads
    source1 = np.repeat((1-alpha) * (rc_solicitation['internal_load_occ'] + rc_solicitation['internal_load_solar']), int(3600/delta))
    source2 = np.repeat(alpha * (rc_solicitation['internal_load_occ'] + rc_solicitation['internal_load_solar']), int(3600/delta))
    source3 = np.repeat(rc_solicitation['external_load_solar'], int(3600/delta))
    t_out = np.repeat(rc_solicitation['t_out'], int(3600/delta))
    t_set_winter = np.repeat(rc_solicitation['t_set_winter'], int(3600/delta))
    t_set_summer = np.repeat(rc_solicitation['t_set_summer'], int(3600/delta))
    
    #Initial values of ti (indoor temperature), tw (walls temperature) and powers
    ti = t_set_winter[0]
    tw = t_set_winter[0]
    p_heat = 0
    p_cold = 0
       
    #Lists to store values of interest 
    heating_need = [0] #watt
    cooling_need = [0] #watt
    t_in = [ti] # indoor temperature [°C]
    t_wall = [tw] # wall temperature [°C]
    print(source1.shape)
    for i in range(1,int(8760*3600/delta)): #loop over time (one year), Euler explicit for resolution (stable under condition !)
        # computation at the current time step
        ts = (ti / ri + tw / rs + source2[i]) / (1 / ri + 1 / rs)
        th = (tw / rw + t_out[i] / re + source3[i]) / (1 / rw + 1 / re)
        # Crank-Nicolson scheme only for nodes linked with a capacity as in ISO 13790. Other 
        # temperatures are constant during the time step. Thus 'semi-semi-explicit' resolution.

        tw_next = (tw + delta/(2*cw) * ((2*th - tw) / rw + (2*ts - tw)/rs)) /\
                  (1 + delta/(2*cw) * (1/rw + 1/rs))
        
        # instead of 2*f[i] one could implement f[i] + f[i+1], small difference
        ti_next = (ti + delta/(2*ci) * ((2*ts - ti)/ri + (2*t_out[i] - ti)/rg + (2*t_out[i] - ti)/rv + 2*source1[i] + 2*p_heat + 2*p_cold))/\
                  (1 + delta/(2*ci) * (1/ri + 1/rg + 1/rv))

        p_heat = ci * (t_set_winter[i] - ti_next) / delta + (ti_next - ts) / ri + (ti_next - t_out[i]) / rg + (
                    ti_next - t_out[i]) / rv - \
                 source1[i]
        
        # p_heat/p_cold are based on a explicit formulation.
        p_heat = np.min([np.max([0, p_heat]), p_heat_max[i]])
        p_cold = ci * (t_set_summer[i] - ti_next) / delta + (ti_next - ts) / ri + (ti_next - t_out[i]) / rg + (
                    ti_next - t_out[i]) / rv - \
                 source1[i]
        p_cold = np.max([np.min([0, p_cold]), p_cold_max[i]])

        heating_need.append(p_heat)
        cooling_need.append(p_cold)
        t_in.append(ti_next)
        t_wall.append(tw_next)

        # replace for next time step
        tw = tw_next
        ti = ti_next
    
    return (heating_need, cooling_need, t_in, t_wall) 
    

(heating_need, cooling_need, t_in, t_wall) = R6C2(simu_parameters, rc_solicitation, rc_parameters)
(heating_need_crank, cooling_need_crank, t_in_crank, t_wall_crank) = R6C2_crank_nicolson(simu_parameters, rc_solicitation, rc_parameters)

#%% print some figures
plt.figure('temperature')
plt.plot(np.repeat(t_set_winter,int(3600/delta)))
plt.plot(np.repeat(t_set_summer,int(3600/delta)))
plt.plot(np.repeat(t_out,int(3600/delta)))
plt.plot(t_in)
plt.plot(t_wall)
plt.xlabel('time step')
plt.ylabel('temperature (°C)')
plt.legend(['t_set_winter','t_set_summer','t_out','t_in', 't_wall'])

plt.figure('power')
plt.plot(np.repeat(internal_load_occ,int(3600/delta)))
plt.plot(np.repeat(internal_load_solar,int(3600/delta)))
plt.plot(heating_need)
plt.plot(cooling_need)
plt.xlabel('time step')
plt.ylabel('needs (W)')
plt.legend(['internal_load_occ','internal_load_solar','heating_need','cooling_need'])

#Annual heating needs calculation (MWh)
annual_heating_need_ref = np.mean(heating_need)*8760/1e6
print('Annual heating needs (MWh): ', annual_heating_need_ref)
#Annual cooling needs (MWh)
annual_cooling_need_ref = np.mean(cooling_need)*8760/1e6
print('Annual cooling needs (MWh): ',annual_cooling_need_ref)
# Daily heating needs (Wh)
daily_heating_need_ref = heating_need.reshape(365, int(24*3600/delta)).mean(1)


#%% Evaluation of a retrofit strategy : wall insulation
print("-----Evaluation of a retrofit strategy : wall insulation-----")
rc_parameters_retrofit = rc_parameters.copy()
rc_parameters_retrofit['r_cond_wall'] = 1/((1)*s_out) # u_out = 1 after retrofit strategy
(heating_need_retrofit, cooling_need_retrofit, t_in_retrofit, t_wall_retrofit) = R6C2(simu_parameters, 
                                                                    rc_solicitation, rc_parameters_retrofit)


annual_heating_need_retrofit = np.mean(heating_need_retrofit)*8760/1e6
annual_cooling_need_retrofit = np.mean(cooling_need_retrofit)*8760/1e6
print('Strategy decrease heating needs by: ' , np.round((annual_heating_need_ref-annual_heating_need_retrofit)/annual_heating_need_ref*100,1), '%' )
print('Strategy decrease cooling needs by (!!!): ' , np.round((-annual_cooling_need_ref+annual_cooling_need_retrofit)/annual_heating_need_ref*100,1), '%' )
#TODO change ventilation strategy to reduce cooling needs

#%% Parameter identification from daily energy measurment (10th to 21th first day of the year)
print("-----modele calibration from measured data-----")
daily_heating_need_measured = np.array([7464, 7321, 7219, 7077, 6975, 6833, 6722, 
                                        6610, 6469, 6349, 6248])
# We want to indentify 'r_cond_wall' to match measured data with simulated data
plt.figure('identification')
plt.plot(daily_heating_need_measured, '*k')
plt.ylabel('daily_heating_need (Wh)')
plt.ylabel('time (day)')
possible_r_cond_wall = np.linspace(rc_parameters['r_cond_wall'], rc_parameters['r_cond_wall']/3,20)
rc_parameters_temp = rc_parameters.copy()
resu_rmse = []
resu_identification = []
for i in possible_r_cond_wall:
    rc_parameters_temp['r_cond_wall'] = i
    (heating_need_temp, cooling_need_temp, t_in_temp, t_wall_temp) = R6C2(simu_parameters, 
                                                                    rc_solicitation, rc_parameters_temp)
    daily_heating_need_temp = heating_need_temp.reshape(365, int(24*3600/delta)).mean(1)[10:21]
    print('r_cond_wall = ', str(i).format('E') )
    rmse = (np.mean((daily_heating_need_measured - daily_heating_need_temp)**2))**0.5
    print('RMSE = ', np.round(rmse,1))
    resu_identification.append(i)
    resu_rmse.append(rmse)
    plt.plot(daily_heating_need_temp)
r_cond_wall_id = resu_identification[np.argmin(resu_rmse)]
print('identified value of r_cond_wall: ' , r_cond_wall_id )
#TODO identify two parameters instead of one.


#%% stop the heating system every evening beween 7 p.m and 9 p.m (2 hours) to reduce the stress on the network
#what is the impact on indoor comfort ?

#%% Store energy during afternoon hours (2 p.m to 3 p.m)  by optimizing temperature setpoint (or power?)
# We have a specific constraint: it is not possible to lower the temperature setpoint, we can only increase it


