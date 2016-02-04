from __future__ import print_function
from __future__ import division

import caffe
import numpy as np
from pdb import set_trace
from random import random


# A layer that crops/flips both data and labels in the same way.
class CropFlip(caffe.Layer):
    def setup(self, bottom, top):
        self.imageSize = (384,384)
        assert(len(top) == 2)
        assert(len(bottom) == 2)

    def reshape(self, bottom, top):
        assert( tuple(bottom[0].shape[2:]) == tuple(bottom[1].shape[2:]) )
        images = bottom[0].shape[0]
        top[0].reshape( images,3,self.imageSize[0],self.imageSize[1] )
        top[1].reshape( images,1,self.imageSize[0],self.imageSize[1] )

    def forward(self, bottom, top):
        top[0].data[...] = 0
        top[1].data[...] = 255

        h = bottom[0].shape[2]
        w = bottom[0].shape[3]

        sz = self.imageSize

        scale = max(h/sz[0], w/sz[1])
        scale = scale * (1 + (random()-.5)/5)

        sy = np.round(scale * (np.arange(0,sz[0] - sz[0]/2 + h/2) )).astype(int)
        sx = np.round(scale * (np.arange(0,sz[1] - sz[1]/2 + w/2) )).astype(int)

        sy = sy[:sz[0]]
        sx = sx[:sz[1]]

        mirror = 1
        if random() > 0.5:
            sx = sx[::-1]

        #There is a more numpy way to do this..
        #mirror = -1
        #slice_y = slice(sy[oky][0],sy[oky][-1])
        #slice_x = slice(sx[okx][0],sx[okx][-1],mirror)

        okx = np.logical_and( 0 <= sx, sx < w)
        oky = np.logical_and( 0 <= sy, sy < h)

        top[0].data[:,:,:np.sum(oky),:np.sum(okx)] = bottom[0].data[:,:,sy[oky],:][:,:,:,sx[okx]]
        top[1].data[:,:,:np.sum(oky),:np.sum(okx)] = bottom[1].data[:,:,sy[oky],:][:,:,:,sx[okx]]

        #r = np.random.random( top[1].data.shape) < 4./5
        #top[1].data[ np.logical_and(top[1].data == 0,r)] = 255

        '''
        if np.sum(top[1].data) == 0:
            import matplotlib.pyplot as plt
            tmp = bottom[0].data[:,:,sy[oky],:]
            tmp = tmp[0]
            tmp = tmp.transpose(2,1,0)
            set_trace()
            plt.imshow(tmp)
            plt.show()
            print("in layer")
        '''

import json
import lmdb

# A layer that crops/flips both data and labels in the same way.
class CropFlipData(caffe.Layer):
    def setup(self, bottom, top):
        assert(len(top) >  1)
        assert(len(bottom) == 0)
        params = json.loads(self.param_str)
        self.batch_size = params["batch_size"]
        self.imageSize = (params["crop_size"],params["crop_size"])

        try:
            self.sample_rate = params["sample"]
        except KeyError:
            self.sample_rate = 1
        #Now setup DBs
        self.image_txn = lmdb.Environment(params["image_db"]).begin(write=False)
        self.label_txn = lmdb.Environment(params["label_db"]).begin(write=False)

        db_size_image = self.image_txn.stat()["entries"]
        db_size_label = self.label_txn.stat()["entries"]
        assert(db_size_image == db_size_label)

        self.image_crs = self.image_txn.cursor().iternext()
        self.label_crs = self.label_txn.cursor().iternext()

        self.image_datum = caffe.proto.caffe_pb2.Datum()
        self.label_datum = caffe.proto.caffe_pb2.Datum()

        self.epoch = 0
        self.count = 0

        #for some reason StopIteration takes ages to get thrown
        #self.db_size = db_size_image
        #self.db_pos = 0
        #if self.db_pos <  self.db_size:

    def reshape(self, bottom, top):
        images = self.batch_size
        top[0].reshape( images,3,self.imageSize[0],self.imageSize[1] )
        top[1].reshape( images,1,self.imageSize[0],self.imageSize[1] )

        if len(top) > 2:
            top[2].reshape( 1 )

    def forward(self, bottom, top):
        print("count",self.count)
        self.count += 1

        for i in range(self.batch_size):
            #image loop
            try:
                image_key,image_value = next(self.image_crs)
                label_key,label_value = next(self.label_crs)
                #self.db_pos += 1
            except StopIteration:
                # The start of each epoch must align with the batch start
                assert(i == 0)
                self.epoch += 1
                print("Epoch:", self.epoch,"batch position",i)
                # reset cursor to beginning
                self.image_crs = self.image_txn.cursor().iternext()
                self.label_crs = self.label_txn.cursor().iternext()
                image_key,image_value = next(self.image_crs)
                label_key,label_value = next(self.label_crs)

            self.image_datum.ParseFromString(image_value)
            self.label_datum.ParseFromString(label_value)
            image_array = caffe.io.datum_to_array(self.image_datum).astype(np.float32)

            #mean subtraction
            image_array[0] -= 104.00699
            image_array[1] -= 116.66877
            image_array[2] -= 122.67892

            label_array = caffe.io.datum_to_array(self.label_datum)

            assert( tuple(image_array.shape[1:]) == tuple(label_array.shape[1:]) )

            h = image_array.shape[1]
            w = image_array.shape[2]
            sz = self.imageSize

            scale = max(h/sz[0], w/sz[1])
            scale = scale * (1 + (random()-.5)/5)

            sy = np.round(scale * (np.arange(0,sz[0] - sz[0]/2 + h/2) )).astype(int)
            sx = np.round(scale * (np.arange(0,sz[1] - sz[1]/2 + w/2) )).astype(int)

            sy = sy[:sz[0]]
            sx = sx[:sz[1]]

            mirror = 1
            if random() > 0.5:
                sx = sx[::-1]

            okx = np.logical_and( 0 <= sx, sx < w)
            oky = np.logical_and( 0 <= sy, sy < h)

            top[0].data[i,...] = 0
            top[1].data[i,...] = 255
            top[0].data[i,:,:np.sum(oky),:np.sum(okx)] = image_array[:,sy[oky],:][:,:,sx[okx]]
            top[1].data[i,:,:np.sum(oky),:np.sum(okx)] = label_array[:,sy[oky],:][:,:,sx[okx]]

        if len(top) > 2:
            top[2].data[0] = self.epoch

        #r = np.random.random( top[1].data.shape) < 4./5
        #top[1].data[ np.logical_and(top[1].data == 0,r)] = 255
        if self.sample_rate < 1:
            r = np.random.random( top[1].data.shape) < self.sample_rate
            top[1].data[ np.logical_and(top[1].data != 0,r)] = 0


# A layer that crops/flips both data and labels in the same way.
class EpochAccum(caffe.Layer):
    def setup(self, bottom, top):
        assert(len(bottom) > 1)
        # Offset self.iteration to -1 so that it starts with 1
        self.iteration = -1
        self.epoch = 0

        # How many iteration to collect before printing output
        self.iter_per_epoch = 18
        self.print_now = False

    def reshape(self, bottom, top):
        # Implict assumption that reshape is called only once per iteration
        # by caffe.
        self.iteration += 1

        top[0].reshape(2)
        if self.iteration % self.iter_per_epoch == 0:
            self.print_now = True
        else:
            self.print_now = False
            top[0].reshape(0)

        print(self.iteration)

    def forward(self,bottom,top):
        #print("ec",self.epoch_change)
        if  self.print_now:
            top[0].data[0] = self.epoch
            top[0].data[1] = self.iteration
            self.print_now = False
            self.print_next = False
