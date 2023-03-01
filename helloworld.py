# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:46:38 2018

@author: lucia
"""

import time as t
import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from scipy.interpolate import interp1d
from nidaqmx.constants import AcquisitionType
from nidaqmx.constants import TerminalConfiguration

FREC=500 #frecuencia de la señal a muestrear
fSampling = 450.0 #10000, 1000, 450 Hz
sampling_time=200/FREC #5, 50, 200 periodos
nSamples =int(fSampling*sampling_time) #samples
task=nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan('Dev1/ai0',terminal_config=TerminalConfiguration.DIFFERENTIAL,min_val=-10,max_val=10)
task.timing.cfg_samp_clk_timing(fSampling,sample_mode=AcquisitionType.FINITE)
t1=t.time()
valores = task.read(number_of_samples_per_channel=nSamples)
t2=t.time()
task.stop()
task.close()
# la información temporal no la genera la DAQ, hay que porporcionarla: timestamp
timestamp=np.arange(len(valores))*(t2-t1)/len(valores)
plt.figure()
plt.plot(timestamp,valores) #grafica 1
fourier=np.fft.fftshift(np.fft.fft(valores))
freq=np.arange(len(valores))*fSampling/len(valores)
freq=freq-freq[int(len(freq)/2)]
plt.figure()
plt.plot(freq,abs(fourier)) #grafica 2
# test de muestreo instantáneo vs muestreo retenido
k1=np.zeros([2*len(valores)])
k2=np.zeros([2*len(valores)])
for i in range(len(k1)):
    if i%2:
        k1[i]=valores[int(i/2)]
        k2[i]=valores[int(i/2)]
        k2[i-1]=valores[int(i/2)]



fourier2=np.fft.fftshift(np.fft.fft(k1))
fourier3=np.fft.fftshift(np.fft.fft(k2))
#duplicamos la frecuencia de muestreo de forma ficticia para ajustar al doble de muestras, reproduciendo muestreo instantáneo y retenido
freq=np.arange(len(k1))*(2*fSampling)/len(k1)
freq=freq-freq[int(len(freq)/2)]
plt.figure()
plt.plot(freq,abs(fourier2),freq, abs(fourier3)) #grafica 3
# Aplicamnos la función spline para recuperar las señales
tiempos=np.arange(1000)*timestamp[len(timestamp)-1]/999 # 1000 puntos a partir de 100
interpolado=interp1d(timestamp,valores,kind='cubic') #interpolacion cubica: interpolado es la funcion
plt.figure()
plt.plot(tiempos,interpolado(tiempos)) #grafica 4