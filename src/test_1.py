from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import itertools
import copy
import argparse
import facenet
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from glob import glob

def main(args):
    sess = init_graph(args.model)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    
    true = 0
    false = 0

    df = pd.read_csv('../relations.csv')
    c_num = list(df['child'])
    p_num = list(df['parent'])
        
    child = glob('test_org/*/*.jpg')
    child += glob('test_org/*/*.PNG')
    child += glob('test_org/*/*.png')
    parents = glob('../missing_child/father/*.jpg')
    parents+= glob('../missing_child/mother/*.jpg')
#    child = child[:30]
    child.sort()
    parents.sort()
#    print(child)
    for i in range(len(child)):
        child_num = p_num[c_num.index(int(child[i].split('/')[1]))]
        child_num = str(child_num)+'.jpg'
        #child_num = child[i].split('/')[-1]
        val_list = []
        for j in range(len(parents)):
            images = read_img(child[i], parents[j], args.image_size)
            curr_val = get_val(images, args, sess, images_placeholder, embeddings, phase_train_placeholder)

            if curr_val==-1:
                print("Illegible face")
            val_list.append(curr_val)

        val_list = np.array(val_list)
        sor_ind = np.argsort(val_list)
        sor_list = []

        for j in range(5):
            im_path = parents[int(sor_ind[j])]
            sor_list.append(im_path.split('/')[-1])

        if child_num in sor_list: true = true+1
        else: false = false+1
        print(i, '/', len(child))
    print('Total number of test cases = ', len(child), '\nTrue = ', true, '\nFalse = ', false)

def read_img(img_1, img_2, img_size):
    imgs = np.zeros((2, img_size, img_size, 3))

    im = Image.open(img_1).convert('RGB')
    im = im.resize((img_size, img_size))
    im = np.array(im)
    imgs[0] = facenet.prewhiten(im)

    im = Image.open(img_2).convert('RGB')
    im = im.resize((img_size, img_size))
    im = np.array(im)
    imgs[1] = facenet.prewhiten(im)

    return imgs        

def init_graph(model):
    sess = tf.InteractiveSession()
    facenet.load_model(model)
    return sess

def get_val(img, args, sess, images_placeholder, embeddings, phase_train_placeholder):
        
    if(len(img) != 2):
        return -1
    # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: img, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict)

    #nrof_images = len(img)
    sub = np.subtract(emb[0,:], emb[1,:])
    dist = np.sqrt(np.sum(np.square(sub)))  
    return dist

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
