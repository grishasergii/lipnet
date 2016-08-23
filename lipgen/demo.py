from liposome import LiposomeUnilamellar, LiposomeMultilamellar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from helpers import prepare_dir
import os

w = 128
h = 128
out_dir = './output/synthetic_examples'
prepare_dir(out_dir)

for i in xrange(100):
    print i
    liposome = LiposomeMultilamellar(w, h)
    img_path = os.path.join(out_dir, 'liposome_{}.png'.format(i))
    mpimg.imsave(img_path, liposome.data, cmap='Greys_r', vmin=0, vmax=1)

#imgplot = plt.imshow(liposome.data, vmin=0, vmax=1)
#imgplot.set_cmap('Greys_r')
#plt.show()
print 'finish'
