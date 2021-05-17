Created on Wed Mar 17 14:18:24 2021

@author: thomas.berthou
"""
import matplotlib.pyplot as plt
import numpy as np
R_mur = 5.62e-4        #K/W
C_mur = 162251798.02        #[J/K]
C_int = 23063797.26*3  #[J/K]
R_int = 8.10491e-5 #K/W
R_ven =    1.56e-4 #K/W

delta = 3600
te = [9,8,7,6,5,4,5,5,6,6,7,7,8,8,9,10,11,12,12,11,11,10,10,9]*10
ti = te[0]
tm = np.mean(te)
save_ti = [ti]
save_tm = [tm]
for i in range(1,len(te)) :
    tm = ((ti-tm)/R_int + (te[i]-tm)/R_mur) * delta/C_mur + tm
    ti = ((tm-ti)/R_int + (te[i]-ti)/R_ven) * delta/C_int + ti
    save_ti.append(ti)
    save_tm.append(tm)

plt.figure(1)
plt.plot(save_ti, 'k')
plt.plot(te, 'b')
plt.plot(save_tm, 'r')
plt.legend(['ti', 'te', 'tm']) 
