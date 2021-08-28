"""Performs face alignment and calculates L2 distance between the embeddings of images."""


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
import align.detect_face
import matplotlib.pyplot as plt
#plt.switch_backend('TkAgg')


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#detector = dlib.get_frontal_face_detector()
#imgs.append(misc.imread("../lag_align/leonardo_dicaprio/0fd9487c5de497ea0772a95a280_2.png"), mode='RGB')
#imgs.append("../lag_align/robert_downey_jr/The_Avengers_00438_2.png")
#imgs.append("/home/admin2/grp6/facenet/datasets/frames/temp.jpg")

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

def main(args):
    writer = None
    prev_name = " "
    prev_avg = 0
    last_frame = None
    with open(args.child_path) as f:
        child_paths = f.readlines()
    child_paths = [x.strip() for x in child_paths]
    assert len(child_paths)>0
    curr_sess = init_graph(args.model)
    (pnet, rnet, onet) = init_mtcnn(args.gpu_memory_fraction)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    red = 0
    t_face = 0
    for cpath in child_paths:
        cert_avg = 0
        k = 0
        curr_positive = 0
        count = 0
        continuity = 0
        red_list = []
        cap = cv2.VideoCapture(args.video_in)
        child_image = process_child(cpath, pnet, rnet, onet,
                                args.margin, args.image_size)
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == True and count%4==0:
                    print('Read %d frame: ' % count, ret)
                    val_list = []
                    bb, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    frame_size = np.asarray(frame.shape)[0:2]
                    min_val = 10.0
                    max_val = 0.0
                    min_x, min_y, min_h, min_w = 0, 0, 0, 0
                    for (i, rect) in enumerate(bb):
                        t_face += 1
                        imgs = []
                        imgs.append(child_image)
                        (x, y, w, h) = rect_to_bb(rect, args.margin, frame_size)
                        cropped = frame[y:y+h, x:x+w]
                        try:
                            aligned = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                        except ValueError:
                            continue
                        prewhitened = facenet.prewhiten(aligned)
                        imgs.append(prewhitened)
                        images = np.stack(imgs)
                        #cv2.imwrite("../datasets/frames/temp.jpg", crop_img)
                        
                        curr_val = get_val(images, args, curr_sess, images_placeholder, embeddings, phase_train_placeholder)
                        print(curr_val)
                        if curr_val==-1:
                            print("Illegible face")
                        else:
                            if curr_val<min_val:
                                min_x = x
                                min_y = y
                                min_h = h
                                min_w = w
                                min_val = curr_val
                            if curr_val>max_val:
                                max_val = curr_val
                            val_list.append(curr_val)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        #cv2.putText(frame, str(curr_val), (x - 10, y - 10), 
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    #cv2.imwrite(os.path.join("../datasets/frames", "frame{:d}.jpg".format(count)), frame)
                    #print(frame.shape)
                    
                    green = (sum(val_list)-min_val)/(len(val_list)-1)
                    cert = abs(green-min_val)/abs(max_val-min_val)
                    #plt.scatter(val_list, [1.0]*len(val_list))
                    #plt.show()
                    #Using new formula
                    #val_list.sort()
                    #for i in range(len(val_list)):
                    #    val_list[i] = val_list[i] - min_val
                    #mean = sum(val_list)/len(val_list)
                    #print("Mean : {}".format(mean))
                    #cert = abs(mean-min_val)/abs(max_val-min_val)
   
                    k = k + 1
                    if cert>0.50:
                        #red_list.append(min_x**2+min_y**2)
                        red += 1
                        curr_positive += 1
                        cert_avg = cert_avg + cert
                        cv2.rectangle(frame, (min_x, min_y), (min_x + min_w, min_y + min_h), (0, 0, 255), 2)
                        cv2.putText(frame, "{0:.3f}".format(cert), (min_x - 10, min_y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        continuity += 1  
                    else:
                        if continuity<=3:
                            red -= continuity
                        continuity = 0
                    print("Certainty: {0:.3f}".format(cert))
                    cv2.putText(frame, "Frame: "+str((int)(count/4)), (0, 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "Current Test: "+os.path.basename(os.path.dirname(cpath)), (0, 35), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "Avg Certainty for "+prev_name+" : "+str(prev_avg), (0, 55), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if writer is None:
                        writer = cv2.VideoWriter(args.video_out,fourcc, 10.0, 
                                                 frame.shape[::-1][1:3])
                        wframe = frame.shape[::-1][1]
                        hframe = frame.shape[::-1][2]
                     
                    writer.write(frame)
                    count += 1
            else:
                    if ret==False:
                        break
                    count += 1
        if k!=0:        
            cert_avg = cert_avg / curr_positive
        else:
            print("Subject not found!")
        #Remove deviating points
        #rmean = np.mean(red_list, axis=0)
        #rstd = np.std(red_list, axis=0)
        #red_list = [x for x in red_list if x<rmean+2*rstd]
        #red_list = [x for x in red_list if x>rmean-2*rstd]
        #red += len(red_list)
        print("Average certainty : {}".format(cert_avg))
        print("Total frames : {}".format(k))
        last_frame = np.zeros((hframe,wframe,3), np.uint8)
        cv2.putText(last_frame, "Avg Certainty for "+os.path.basename(os.path.dirname(cpath))+" : "+str(cert_avg), (0, 55), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i in range(2):
            writer.write(last_frame)
        prev_name = os.path.basename(os.path.dirname(cpath))
        prev_avg = cert_avg;
        cap.release()
    
    print("Red Faces: {}".format(red))
    print("Total Faces: {}".format(t_face))
    cv2.destroyAllWindows()
    
def process_child(img_path, pnet, rnet, onet, margin, image_size):
    print("Detecting child face in "+os.path.basename(os.path.dirname(img_path))+"...")
    img = misc.imread(os.path.expanduser(img_path), mode='RGB')
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        print("can't detect child face", img_path)
        exit(0)
    det = np.squeeze(bounding_boxes[0,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = facenet.prewhiten(aligned)
    return prewhitened
    
    
def init_graph(model):
    sess = tf.InteractiveSession()
    facenet.load_model(model)
    return sess

def init_mtcnn(gpu_memory_fraction):
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, 'align')
    return (pnet, rnet, onet)

def get_val(img, args, sess, images_placeholder, embeddings, phase_train_placeholder):
        
    #images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    # Get input and output tensors

    #images = load_and_align_data(img, args.image_size, args.margin, args.gpu_memory_fraction)
    if(len(img) != 2):
        return -1
    # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: img, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict)

    #nrof_images = len(img)
    sub = np.subtract(emb[0,:], emb[1,:])
    dist = np.sqrt(np.sum(np.square(sub)))  
    return dist;
            
            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def rect_to_bb(rect, margin, img_size):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    '''x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y'''
    
    x = rect[0]-margin/2
    y = rect[1]-margin/2
    w = rect[2]+margin/2 - x
    h = rect[3]+margin/2 - y

    # return a tuple of (x, y, w, h)
    return (int(x), int(y), int(w), int(h))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('child_path', type=str)
    parser.add_argument('video_in', type=str)
    parser.add_argument('video_out', type=str)
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
