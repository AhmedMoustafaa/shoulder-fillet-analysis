import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from math import sqrt

data = pd.read_csv('final-data.csv.csv')
r = np.array(data['r'])
r = r.astype(float)
max = np.array(data['v1'])
max = max.astype(float)
x0 = r/0.03
nom = (10*16)/(3.14*(0.03)**3)
k = max/nom

def func(x, a, b, c, d):
    return d + ((a-d)/(1+(x/c)**b))

popt, pcov = curve_fit(func, x0, k)
x1 = np.linspace(0.000001, 0.3, 100)
y = func(x1, *popt)

r1 = np.linspace(0.0037,0.009,1000)

th = pd.read_csv('theoritical_data.csv')
xth = np.array(th['r'])
yth = np.array(th['v'])
popt1, pcov1 = curve_fit(func, xth, yth)
yth1 = func(x1, *popt1)
print(popt1)

plt.figure()
plt.xlabel(r'$\frac{R}{D1}$')
plt.ylabel('K')

plt.scatter(x0,k,color='red', label='collected points',alpha=1)
plt.plot(x1, y, '-r', label='fitted line')

plt.scatter(xth,yth, color='blue',alpha=1, label='theoritical points')
plt.plot(x1,yth1,'-b', label='theoritical fitted-line')

plt.ylim([1,2])
plt.xlim([0.0,0.3])
plt.grid()
plt.legend()
plt.savefig('plot')
plt.show()
