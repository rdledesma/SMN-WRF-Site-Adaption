import pandas as pd

d = pd.read_csv('full_timeline_AR_multi_modelo.csv')
d['valid_time'] = pd.to_datetime(d.valid_time)

import matplotlib.pyplot as plt

d = d[d.corrida == 0]

plt.figure()
plt.plot(d.valid_time, d.GHI_Wm2, '.-b')
plt.plot(d.valid_time, d.ghi, '.-r')
plt.show()