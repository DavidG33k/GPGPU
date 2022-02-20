import matplotlib.pyplot as plt
import numpy as np

temp = []

##############################################################

#peak performance [GFLOPS]
peak_perf = 4.612
#sustainable bandwidth [GB/sec]
stream_bw = 224.32
#[operational intensity, gflops, color, name]
kernel = [[1.7, 4.196, 'red', 'Basic Kernel'], [39.55, 4.061, 'blue', 'Tiled kernel']]

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
