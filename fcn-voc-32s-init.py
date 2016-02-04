from __future__ import division
from __future__ import print_function
import caffe
import numpy as np

from pdb import set_trace

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print( 'input + output channels need to be the same' )
            raise
        if h != w:
            print( 'filters need to be square' )
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt



# init
caffe.set_mode_cpu()

base_model = 'VGG_ILSVRC_16_layers_deploy.prototxt'
base_weights = 'VGG_ILSVRC_16_layers.caffemodel'
base_net = caffe.Net(base_model, base_weights, caffe.TRAIN)

voc_model = 'fcn-voc-32s-train.prototxt'
voc_weights = 'fcn-voc-32s-train.caffemodel'  # where result is going to go
voc_net = caffe.Net(voc_model, caffe.TEST)

# Source and destination paramteres, these are the same because the layers
# have the same names in base_net, voc_net
src_params = ['fc6', 'fc7']  # ignore fc8 because it will be re initialized
dest_params = ['fc6', 'fc7']

# First: copy shared layers
shared_layers = set(base_net.params.keys()) & set(voc_net.params.keys())
shared_layers -= set(src_params + dest_params)

for layer in sorted(list(shared_layers)):
    print("Copying shared layer",layer)
    voc_net.params[layer][0].data[...] = base_net.params[layer][0].data
    voc_net.params[layer][1].data[...] = base_net.params[layer][1].data

# Second: copy over the fully connected layers
# fc_params = {name: (weights, biases)}
fc_params = {}
for pr in src_params:
    fc_params[pr] = (base_net.params[pr][0].data, base_net.params[pr][1].data)

# conv_params = {name: (weights, biases)}
conv_params = {}
for pr in dest_params:
    conv_params[pr] = (voc_net.params[pr][0].data, voc_net.params[pr][1].data)

for pr, pr_conv in zip(src_params, dest_params):
    print('(source) {} weights are {} dimensional and biases are {} dimensional'\
      .format(pr, fc_params[pr][0].shape, fc_params[pr][1].shape))
    print('(destn.) {} weights are {} dimensional and biases are {} dimensional'\
      .format(pr_conv, conv_params[pr_conv][0].shape, conv_params[pr_conv][1].shape))

    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]


# Third: inititalize upsampling
interp_layers = [k for k in voc_net.params.keys() if 'up' in k]
# do net surgery to set the deconvolution weights for bilinear interpolation
interp_surgery(voc_net, interp_layers)

#Finally: Save resulting network
voc_net.save(voc_weights)
