from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from keras.models import load_model


#FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.3):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss

"""
########################### Initiate FR model  ######################################
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)
print("Facenet_model_compiled")

FRmodel.save('FR_net.h5')
"""

########################################## Load model ############

FRmodel = load_model('FR_net.h5', custom_objects={'triplet_loss': triplet_loss})
print("Model loaded")



##################################image database of peps  ###############################
def prepare_database():
    database = {}

    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)

    return database



def face_recognizer(img,database):
   
    #vc = cv2.VideoCapture(0)
    im = cv2.imread(img)
    who_is_it(im, database, FRmodel)



def who_is_it(image, database, model):
    
    encoding = img_to_encoding(image, model)
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' %(name, dist))


if __name__ == "__main__":
    database = prepare_database()
    face_recognizer('out4.jpg',database)

