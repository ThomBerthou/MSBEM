# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:40:16 2021
@author: thomas.berthou
The Most Simple Building Energy Model is for students and teachers to learn and teach about energy building simulation through a simple exemple
This model simulate heating and cooling needs for a buildings at a chosen time step (maximum is 1800 secondes)
What to do with this model :
 - Load shedding assesment
 - Sensitivity analysis
 - Refurbishing assesment
 - Calibration from on site measurment or detail simulation
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
s_in = 435 # opaques walls surface (vertical and horizontal) in contact with outdoor temperature [m²]
s_out = 235 # opaques walls surface (vertical and horizontal) in contact with indoor temperature [m²]
s_windows = 15 # surface of windows [m²]
u_out = 1.5 # mean U value of opaques walls (vertical and horizontal) [W/(K.m²)]
u_windows = 2.5 #mean U value of windows (conduction and convection included) [W/(K.m²)]
h_in = 6 # indoor convection coefficient [W/(K.m²)]
h_out = 20 # outdoor convection coefficient [W/(K.m²)]
m_air_new = 0.6 # mass flow rate of fresh air [Vol/hour]
v_in = 250 # indoor air volume [m3]
inertia_surf = 432000 # surface inertia of building structure [J/K.m²]
rc_parameters = dict() #dictionary with the R and C values
rc_parameters['r_conv_ext'] = 1/(h_out*s_out)
rc_parameters['r_cond_wall'] = 1/(u_out*s_out)
rc_parameters['r_conv_int'] = 1/(h_in*s_in)
rc_parameters['r_infiltration'] = 1/(rho_air*c_air*m_air_new*v_in/3600)
rc_parameters['r_wondows'] = 1/(u_windows*s_windows)
rc_parameters['C_air'] = v_in * c_air * rho_air * 15 # add inertia of furniture and light walls (x15)
rc_parameters['C_wall'] = inertia_surf * s_floor

#%% Simulation parameters
delta = 600 # simulation time step in seconde (300 secondes to 1800 secondes)
p_heat_max = 100 * s_floor # maximum heat delivered (watt)
p_cold_max = -50 * s_floor # maximum cold delivered (watt)

#%% R6C2 dynamic thermal model (no HVAC system)
def R6C2 (delta, p_heat_max, p_cold_max, rc_solicitation, rc_parameters) :
    '''
    adapted from Berthou et al. 2014 : Development and validation of a gray box model 
    to predict thermal behavior of occupied office buildings, Energy and Buildings
    TODO : more stable discretization scheme, ground temperature, add HVAC systemes... 
    '''
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
    for i in range(1,int(8760*3600/delta)): #loot over time (one year), Euler explicit for resolution (stable under condition !)
        ts = (ti/ri + tw/rs + source2[i])/(1/ri + 1/rs)
        th = (tw/rw + t_out[i]/re + source3[i])/(1/rw + 1/re)
        tw = ((th-tw)/rw + (ts-tw)/rs)*delta/cw + tw
        ti = ((ts-ti)/ri + (t_out[i]-ti)/rg + (t_out[i]-ti)/rv + source1[i] + p_heat + p_cold)*delta/ci + ti
         
        p_heat = ci*(t_set_winter[i]-ti)/delta + (ti-ts)/ri + (ti-t_out[i])/rg + (ti-t_out[i])/rv - source1[i]
        p_heat  = np.min([np.max([0,p_heat]),p_heat_max])
        p_cold = ci*(t_set_summer[i]-ti)/delta + (ti-ts)/ri + (ti-t_out[i])/rg + (ti-t_out[i])/rv - source1[i]
        p_cold = np.max([np.min([0,p_cold]),p_cold_max])
        
        heating_need.append(p_heat)
        cooling_need.append(p_cold)
        t_in.append(ti)
        t_wall.append(tw)
    
    return (heating_need, cooling_need, t_in, t_wall)

(heating_need, cooling_need, t_in, t_wall) = R6C2(delta, p_heat_max, p_cold_max, rc_solicitation, rc_parameters)

#%% print some figures
plt.figure(1)
plt.plot(np.repeat(t_set_winter,int(3600/delta)))
plt.plot(np.repeat(t_set_summer,int(3600/delta)))
plt.plot(np.repeat(t_out,int(3600/delta)))
plt.plot(t_in)
plt.plot(t_wall)
plt.xlabel('time step')
plt.ylabel('temperature (°C)')
plt.legend(['t_set_winter','t_set_summer','t_out','t_in', 't_wall'])

plt.figure(2)
plt.plot(np.repeat(internal_load_occ,int(3600/delta)))
plt.plot(np.repeat(internal_load_solar,int(3600/delta)))
plt.plot(heating_need)
plt.plot(cooling_need)
plt.xlabel('teme step')
plt.ylabel('Needs (W)')
plt.legend(['internal_load_occ','internal_load_solar','heating_need','cooling_need'])
