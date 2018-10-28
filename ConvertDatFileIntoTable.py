
import pandas as pd
import numpy as np
import cmath
import plotly.graph_objs as go
import plotly.plotly as py
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100
N = 1024
data = pd.read_csv('vaikuFT.dat',header=None, delimiter=r"\s+")
x = data[0]
y = data[1]

xlist = []
ylist = []

for i in range(0,N):
   xlist.append(float(x.iloc[i]))
   ylist.append(float(y.iloc[i]))

pi2 = cmath.pi * 2.0


def DFT(fnList):
    N = len(fnList)
    FmList = []
    for m in range(N):
        Fm = 0.0
        for n in range(N):
            Fm += fnList[n] * cmath.exp(- 1j * pi2 * m * n / N)
        FmList.append(Fm / N)
    return FmList


X = DFT(ylist)

powers_all = np.abs(np.divide(X, N//2))
powers = powers_all[0:N//2]
frequencies = np.divide(np.multiply(SAMPLE_RATE, np.arange(0, N/2)), N)

f = open("outputResult.text", "w")

threshold = 0.001
for (i, Fm) in enumerate(X):
    if abs(Fm) > threshold:
        f.write( " Frequencies = " + str(frequencies) +"\n")
        f.write( " Power = " + str(powers)+"\n")

f.close()

_, plots = plt.subplots(2)
plots[0].plot(x)
plots[1].plot(frequencies, powers)
plt.show()


trace1 = go.Scatter(
                    x = xlist,
                    y = ylist,
                    mode = "lines",
                    name = "Freq Audio")

layout = go.Layout(
    showlegend=True
)

trace_data = [trace1]
fig = go.Figure(data=trace_data, layout=layout)
py.plot(fig, filename='sample-audio')
