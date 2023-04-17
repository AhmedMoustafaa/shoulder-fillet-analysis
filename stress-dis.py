import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

df = pd.read_csv('venv/final-probe.csv')
x1 = np.array(df['X'])
y1 = np.array(df['V'])

def func(x,a,b):
    return a*abs(x)+b
print(x1[1:12])
popt, pvoc = curve_fit(func, x1, y1)

x2 = np.linspace(x1[0], x1[-1], 100)
y2 = func(x2, *popt)

plt.figure()

plt.xlabel('X')
plt.ylabel('shear stress')
plt.scatter(x1,y1,color='blue',label='collected points by probe')
plt.plot(x2,y2,'-b', label='fitted line')
plt.legend()
plt.show()