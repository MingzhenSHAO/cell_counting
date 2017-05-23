import caffe
import surgery, score

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = './snapshot/unet.caffemodel'

# init		VGG16_ILSVRC
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'upsample' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/voc_1/test.txt', dtype=str)

for _ in range(1):
    solver.step(60000)
    score.seg_tests(solver, False, val, layer='score')
