# *******************************************************************
import argparse
import sys
import os

from utils import *

if not os.path.exists('outputs/'):
    os.makedirs('outputs/')

net = cv2.dnn.readNetFromDarknet('./cfg/yolov3-face.cfg', './model-weights/yolov3-wider_16000.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def _main():
   
    #cap = cv2.VideoCapture('./samples/0.jpg')
    frame = cv2.imread('./samples/f.jpg')
    frm = frame.copy()	
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Remove the bounding boxes with low confidence
    f = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    print(f[0])
    w=f[0][2]
    h=f[0][3]
    x1 =f[0][0]
    y1 =f[0][1]
    #h= f[0][1]-f[0][3]
    #w= f[0][0]-f[0][2]
	
    roi = frm[y1:y1+h,x1:x1+w]
    #frame = cv2.circle(frame, (x2,y2), 8, (255, 0, 0) , 2)    
    cv2.imwrite('outputs/out1.jpg', roi.astype(np.uint8))


if __name__ == '__main__':
    _main()
