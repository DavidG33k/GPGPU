#coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

temp = []

##############################################################

#peak performance [GFLOPS]
peak_perf = 1700
#sustainable bandwidth [GB/sec]
stream_bw = 70
#[operational intensity, gflops, color, name]
kernel = [[0.50, 20, 'red', 'kernel 1'], [10, 100, 'blue', 'kernel 2'], [40, 1000, 'green', 'kernel 3']]

################################################################

plt.figure()

#plot roofline
x = np.arange(0.01,100,0.01)
left_roof = x * stream_bw
for i in range(len(x)):
temp.append(min(left_roof[i],peak_perf))
y = np.array(temp)
plt.plot(x,y,color='goldenrod')

#plot kernels
for i in range(len(kernel)):

plt.plot(kernel[i][0], kernel[i][1], 'p', color=kernel[i][2], label=kernel[i][3])

#setup
plt.xscale("log")
plt.yscale("log")
plt.grid(which="both")
plt.xlim([0.01,100])
plt.title('Roofine')
plt.xlabel('Operational intensity (flops/byte)')
plt.ylabel('Performance [Gflops/sec]')
plt.legend(loc='upper left')

plt.show()

#Output
#widgets RELATED ARTICLES
#widgets CONTRIBUTION
#This article is contributed by Yukawa and text available under CC-SA-4.0
#2021 - TitanWolf
#CONTACTTERMS OF SERVICEPOLICIES
#%%

import matplotlib.pyplot as plt
import numpy as np

temp = []

##############################################################

#peak performance [GFLOPS]
peak_perf = 4358
#sustainable bandwidth [GB/sec]
stream_bw = 224
#[operational intensity, gflops, color, name]
kernel = [[29, 7, 'red', 'flowsCompStraight'], [7, 6, 'blue', 'WithUpdaStraight'], [36, 7, 'green', 'flowsCompHalo'],[3, 6, 'yellow', 'WithUpdaHalo'],[24, 7, 'purple', 'flowsCompWHalo'],[4, 7, 'pink', 'WithUpdaWHalo']]

################################################################

plt.figure()

#plot roofline
x = np.arange(0.01,100,0.01)
left_roof = x * stream_bw
for i in range(len(x)):
temp.append(min(left_roof[i],peak_perf))
y = np.array(temp)
plt.plot(x,y,color='goldenrod')

#plot kernels
for i in range(len(kernel)):
plt.plot(kernel[i][0], kernel[i][1], 'p', color=kernel[i][2], label=kernel[i][3])

#setup
plt.xscale("log")
plt.yscale("log")
plt.grid(which="both")
plt.xlim([0.01,100])
plt.title('Roofine')
plt.xlabel('Operational intensity (flops/byte)')
plt.ylabel('Performance [Gflops/sec]')
plt.legend(loc='upper left')

plt.show()

#%%