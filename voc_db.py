from __future__ import division,print_function
import os
import re
from pdb import set_trace
import lmdb
import numpy as np
from PIL import Image
from scipy.io import loadmat

import sys
if sys.version_info.major > 2:
    print("protocol buffers needs python2")
    sys.exit()

from caffe.io import caffe_pb2, array_to_datum

import shutil


ignore_label = 255
iter_size = 20
batch_size_test = 1
batch_size_train = 1


def get_train_test_split():
    #As train set: pascal train + sds train + val
    pvc_train_fn = "~/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
    pvc_valdn_fn = "~/data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
    sds_train_fn = "~/data/SDS/train.txt"
    sds_valdn_fn = "~/data/SDS/val.txt"

    d = dict( pvc_train = pvc_train_fn,
              pvc_valdn = pvc_valdn_fn,
              sds_train = sds_train_fn,
              sds_valdn = sds_valdn_fn )

    for k,fn in d.items():
        fn = os.path.expanduser(fn)
        with open(fn) as f:
            d[k] = f.read().splitlines()

    train = set( d["pvc_train"] + d["sds_train"] + d["sds_valdn"])
    test  = set( d["pvc_valdn"] ) - train
    return train, test



def scale_func_factory(scale_by=False,scale_to=False):

    def scale(im,is_label=False):
        if scale_by == False and scale_to == False:
            return im

        elif scale_by and scale_to:
            raise ValueError

        elif scale_by:
            new_size = (int(im.size[0]*scale_by),int(im.size[1]*scale_by))

        elif scale_to:
            new_size = (scale_to, scale_to)


        if is_label:
            scale_mode = Image.NEAREST
            im = im.resize( new_size ,scale_mode)
            im = np.array(im)[np.newaxis,:,:]
        else:
            scale_mode = Image.ANTIALIAS
            im = im.resize( new_size ,scale_mode)
            im = np.array(im)[:,:,::-1].transpose(2,0,1)

        return im

    return scale




def build_db(files,db_fn,with_images=True,with_labels=False,
             trf_func=np.array,pad=0):
    # Where from
    pvc_im_path = "~/data/VOCdevkit/VOC2012/JPEGImages/"
    pvc_sg_path = "~/data/VOCdevkit/VOC2012/SegmentationClass/"
    sds_sg_path = "~/data/SDS/cls"
    
    d = dict( pvc_im = pvc_im_path,
              pvc_sg = pvc_sg_path,
              sds_sg = sds_sg_path)
    #Don't know how to clear lmdb using api
    shutil.rmtree(db_fn)

    # Start
    env = lmdb.Environment(db_fn,map_size=10**12)

    for k,fn in d.items():
        d[k] = os.path.expanduser(fn)
    
    files = sorted(list(files))

    for i in range(len(files)):
        file_fn = files[i]
        print(file_fn,i,"/",len(files))

        if with_images:
          im = Image.open(os.path.join(d["pvc_im"],file_fn + ".jpg"))
          #(366,500,3) -> (3,366,500)
          im = trf_func(im)[:,:,::-1].transpose(2,0,1)
          data = im

        if with_labels:
            sg_fn = os.path.join(d["pvc_sg"],file_fn + ".png")
            if  os.path.isfile(sg_fn):
                sg = Image.open(sg_fn)
                #(366,500) -> (1,366,500)
                sg = trf_func(sg)[:,:,np.newaxis].transpose(2,0,1)
            else:
                sg_fn = os.path.join(d["sds_sg"],file_fn + ".mat")
            
                if os.path.isfile(sg_fn):
                    data = loadmat(sg_fn)
                    sg = Image.fromarray(data["GTcls"][0,0][1])
                    sg = trf_func(sg)[:,:,np.newaxis].transpose(2,0,1)

            if with_images:
                print("Check images here.")
                raise NotImplementedError
                data = np.dstack( (np.array(im)[:,:,::-1],np.array(sg))).transpose(2,0,1)
            else:
                data = sg

        key = str(i).zfill(8) + '_' + file_fn.replace(".png",".jpg")
        datum = array_to_datum(data)
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            cursor.put(key,datum.SerializeToString())
    
    for j in range(pad):
        key = str(i+j).zfill(8) + '_' + "pad"

        print(j,"/",pad)

        # Should be able to put in empty images, but im afraid
        # that will raise errors, do this for now.
        if with_images:
            data = np.zeros( (3,384,384),dtype=np.uint8)
        if with_labels:
            data = np.zeros( (1,384,384),dtype=np.uint8 )
            data[...] = ignore_label

        datum = array_to_datum(data)
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            cursor.put(key,datum.SerializeToString())

def convert_VOC():
    from math import ceil
    # What
    train,test = get_train_test_split()

    test_iter = int(ceil(len(test)/(iter_size*batch_size_test)))
    train_iter = int(ceil(len(train)/(iter_size*batch_size_train)))

    pad_size_test = test_iter * iter_size*batch_size_test - len(test)
    pad_size_train = train_iter * iter_size*batch_size_train - len(train)

    print("Given: iter_size = ",iter_size)
    print("       batch_size (test ) = ",batch_size_test)
    print("       batch_size (train) = ",batch_size_train)
    print("       db size (test ) =",len(test))
    print("       db size (train) =",len(train))
    print()
    print("This script will pad the test  set with:",pad_size_test,"empty images")
    print("This script will pad the train set with:",pad_size_train,"empty images")
    print("So that for 1 epoch: train_iter = ",train_iter)
    print()
    print("Check that: test_iter  = ",test_iter)
    print("            ignore_label = ",ignore_label)
    print()
    raw_input("Press enter to continue")

    # Where to
    db_fn_test = "voc_test"
    build_db(test,db_fn_test,with_labels=False,pad=pad_size_test)

    db_fn_test = "voc_lbl_test"
    build_db(test,db_fn_test,with_images=False,with_labels=True,pad=pad_size_test)

    db_fn_train = "voc_train"
    build_db(train,db_fn_train,with_labels=False,pad=pad_size_train)

    db_fn_train = "voc_lbl_train"
    build_db(train,db_fn_train,with_images=False,with_labels=True,pad=pad_size_train)



'''
def convert_VOC_small():
    # What
    train,test = get_train_test_split()

    # Where to
    #db_fn_train = "voc_std_train"
    #build_db(train,db_fn_train,with_labels=False,scale=.8)

    #db_fn_test = "voc_small_test"
    #build_db(test,db_fn_test,with_labels=False,scale=.8)

    #db_fn_train = "voc_small_lbl_train"
    #build_db(train,db_fn_train,with_images=False,with_labels=True,scale_by=.8)

    #db_fn_test = "voc_small_lbl_test"
    #build_db(test,db_fn_test,with_images=False,with_labels=True,scale_by=.8)
'''

if __name__ == "__main__":
    convert_VOC()

