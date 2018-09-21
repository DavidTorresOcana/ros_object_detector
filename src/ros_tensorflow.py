#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# ROS node libs
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int16
from multiprocessing import Queue, Pool
from cv_bridge import CvBridge, CvBridgeError


# General libs
import numpy as np
import os
import sys
import tensorflow as tf
from keras import backend as K
from keras.models import load_model, Model, model_from_json
import cv2

# Detector libs
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body, yolo_eval

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# GPU settings: Select GPUs to use. Coment it to let the system decide
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def crop_image(img, ratio = 0.56):
    '''Cropping the image with the correct ratio (original ratio) of width/height.
    Returns cropped image.
    '''
    height = img.shape[0]
    width =  img.shape[1]
    curr_AR = float(height)/float(width)

    if curr_AR>ratio: # Crop in height
        new_height = int(width*ratio) #magic number due to original image ratio
        img = img[-new_height:,:] # Crop bottom
    else: # Crop in width. Both sides
        width_diff = width - int(float(height)/ratio)
        img = img[:,int(width_diff/2.):-int(width_diff/2.)] # Crop sides

    return img


class ros_tensorflow_obj():
    def __init__(self):
        # ## Initial msg
        rospy.loginfo('  ## Starting ROS Tensorflow interface ##')

        self.sess = K.get_session()

        # # Model preparation 
        # ## Variables
        # load model
        MODEL_NAME = os.path.dirname(os.path.realpath(__file__))+'/../include/ros_object_detector/yolov2/yolov2_coco_py2.h5'
        try:
            MODEL_NAME = os.path.dirname(os.path.realpath(__file__))+'/../' + rospy.get_param(rospy.get_name()+'/model_path_name')
        except:
            rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/model_path_name'))

        self.yolo_model = load_model(MODEL_NAME,  custom_objects={'backend': K,'tf':tf})
        
        # load classes and anchors
        CLASSES_PATH = os.path.dirname(os.path.realpath(__file__))+'/../include/ros_object_detector/yolov2/coco_classes.txt'
        try:
            CLASSES_PATH = os.path.dirname(os.path.realpath(__file__))+'/../' + rospy.get_param(rospy.get_name()+'/classes')
        except:
            rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/classes'))

        self.class_names = read_classes(CLASSES_PATH)        

        # category_index: a dict containing category dictionaries (each holding category index `id` and category name `name`) keyed by category indices.
        self.category_index = {}
        for id_,name in enumerate(self.class_names):
            self.category_index[id_] = {'id':id_,'name': name}
        
        # Label maps map indices to category names
#         self.category_index = label_map_util.create_category_index(categories)
        ANCHORS_PATH = os.path.dirname(os.path.realpath(__file__))+'/../include/ros_object_detector/yolov2/yolo_anchors.txt'
        try:
            ANCHORS_PATH = os.path.dirname(os.path.realpath(__file__))+'/../' + rospy.get_param(rospy.get_name()+'/anchors')
        except:
            rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/anchors'))

        self.anchors = read_anchors(ANCHORS_PATH)
        
         # Get head of model
        self.yolo_outputs = yolo_head(self.yolo_model.output, self.anchors, len(self.class_names))
        
        # Get models input size
        self.model_image_size =  self.yolo_model.inputs[0].get_shape().as_list()[-3:-1]
       
        # Get classes colors
        self.colors = generate_colors(self.class_names)
        
        # Image Aspect ratio parameters
        self.change_AR = False # Change AR of input image?
        try:
            self.change_AR = rospy.get_param(rospy.get_name()+'/change_AR')
        except:
            rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/change_AR'))
        
        self.AR = 503.0/800.0 # Default GMSL cameras AR
        if self.change_AR:
            try:
                self.AR = rospy.get_param(rospy.get_name()+'/AR')
            except:
                rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/AR'))

        # Get Tensors to evaluate
        #CameraInfo_msg = rospy.wait_for_message('/camera/camera_info',CameraInfo)
        ##self.boxes, self.scores, self.classes = yolo_eval(self.yolo_outputs, np.array([CameraInfo_msg.height,CameraInfo_msg.width],dtype=np.float32))
        image_np = CvBridge().imgmsg_to_cv2(  rospy.wait_for_message('/camera/image',Image)  , "rgb8")

        self.input_img_size = image_np.shape

        if self.change_AR:
            if abs(  float(self.input_img_size[0])/float(self.input_img_size[1]) - self.AR  )>0.04: # Depending on how well the model generalizes
                image_np = crop_image(image_np, ratio = self.AR)
                self.input_img_size = image_np.shape
            else:
                self.change_AR = False # Just in case we did not configure it right, to save time
        print(self.input_img_size)
        ############# WE NEED TO KEEP THE ASPECT RATIO USED ON THE NEXAR TRAINING DATA 720/1280 ON INPUT IMAGES
        self.boxes, self.scores, self.classes = yolo_eval( self.yolo_outputs,
                                                     np.array([self.input_img_size[0],self.input_img_size[1]],dtype=np.float32) )

        
        # # ROS environment setup
        # ##  Define subscribers
        self.subscribers_def()
        # ## Define publishers
        self.publishers_def()
        # ## Get cv_bridge: CvBridge is an object that converts between OpenCV Images and ROS Image messages
        self._cv_bridge = CvBridge()
        # ## Init time counter
        self.now = rospy.Time.now()    
     
        
    # Define subscribers
    def subscribers_def(self):
        subs_topic = '/camera/image'
        # try:
        #     subs_topic = rospy.get_param(rospy.get_name()+'/camera_topic')
        # except:
        #     rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/camera_topic'))
        self._sub = rospy.Subscriber( subs_topic , Image, self.img_callback, queue_size=1, buff_size=2**24)

    # Define publishers
    def publishers_def(self):
        out_img_topic = '/image_objects_detect'
        # try:
        #     out_img_topic = rospy.get_param(rospy.get_name()+'/out_img_topic')
        # except:
        #     rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/out_img_topic'))
        self._img_publisher = rospy.Publisher( out_img_topic , Image, queue_size=0)

    # Camera image callback
    def img_callback(self, image_msg):
       
        # Get image as np
        image_np = self._cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")

        # Preprocess your image
        if self.change_AR: # WE NEED TO KEEP THE ASPECT RATIO USED ON THE TRAINING DATA 720/1280 ON INPUT IMAGES
            image_np = crop_image(image_np, ratio = self.AR)
        image_data = cv2.resize(image_np, dsize=(self.model_image_size[1],self.model_image_size[0]),
                                                     interpolation=cv2.INTER_CUBIC).astype(np.float32)
        image_data = np.expand_dims(image_data/255.0, 0)  # Normalize and Add batch dimension.
        try:
            # # Actual detection
            out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes], feed_dict={self.yolo_model.input: image_data})

            # Visualization of the results of a detection.
    #         # draw_boxes(image_np, out_scores, out_boxes, out_classes, self.class_names, self.colors)
            
    #         print(out_boxes.shape,out_classes.astype(np.int32),out_scores.shape)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                out_boxes,
                out_classes.astype(np.int32),
                out_scores,
                self.category_index )
        
            # Publish the results
            self._img_publisher.publish(self._cv_bridge.cv2_to_imgmsg(image_np, "rgb8"))

            rospy.loginfo("  Publishing inference at %s FPS", 1.0/float(rospy.Time.now().to_sec() - self.now.to_sec()))
            self.now =rospy.Time.now()

        except CvBridgeError as e:
            rospy.logerr(e)    

    # Spin once
    def spin(self):
        rospy.spin()
