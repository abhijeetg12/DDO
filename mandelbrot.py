import numpy as np
import matplotlib.pyplot as plt

closs=np.load('options/tloss.npy')
rng=range(len(closs))

plt.plot(rng, closs)
plt.show()
print(len(closs))

