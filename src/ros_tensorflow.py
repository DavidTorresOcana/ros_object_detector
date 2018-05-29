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
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from multiprocessing import Queue, Pool
from cv_bridge import CvBridge, CvBridgeError

# General libs
import numpy as np
import os
import sys
import tensorflow as tf
import cv2

# Detector libs
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# GPU settings: Select GPUs to use. Coment it to let the system decide
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

class ros_tensorflow_obj():
	def __init__(self):
		# ## Initial msg
		rospy.loginfo('  ## Starting ROS Tensorflow interface ##')
		
		# # Model preparation 
		# ## Variables
		# What model to grun
		MODEL_NAME = os.path.dirname(os.path.realpath(__file__))+'/../include/ros_object_detector/pothole_detector/pothole-graph/ssd-mobilenet-25000-07-03-18'
		
		# Path to frozen detection graph. This is the actual model that is used for the object detection.
		PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
		
		# Get other path to frozen detection graph if specified
		try:
			PATH_TO_CKPT = os.path.dirname(os.path.realpath(__file__))+'/../include/ros_object_detector/' + rospy.get_param(rospy.get_name()+'/tf_model/pb_path')
		except:
			rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/tf_model/pb_path'))
		
		# List of the strings that is used to add correct label for each box.
		PATH_TO_LABELS = os.path.join(os.path.dirname(os.path.realpath(__file__))+'/../include/ros_object_detector/pothole_detector/', 'object-detection.pbtxt')
		
		# Get other label definition if specified
		try:
			PATH_TO_LABELS =  os.path.dirname(os.path.realpath(__file__))+'/../include/ros_object_detector/' + rospy.get_param(rospy.get_name()+'/tf_model/labels_definition')
		except:
			rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/tf_model/labels_definition'))
			
		NUM_CLASSES = 1 # As defined in labels file
		# Get other number of objects if specified
		try:
			NUM_CLASSES = rospy.get_param(rospy.get_name()+'/tf_model/num_objects')
		except:
			rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/tf_model/num_objects'))
			
		# ## Load a (frozen) Tensorflow model into memory.
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
				
		# ## Loading label map
		# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
		label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
		self.category_index = label_map_util.create_category_index(categories)
		# ## Get Tensors to run from model
		self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
		self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
		self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		
		# # Tensorflow Session opening: Creates a session with log_device_placement set to True.
		# ## Session configuration
		config = tf.ConfigProto()
		config.log_device_placement = True
		config.gpu_options.allow_growth = True
		# ## Session openning
		try:
			with detection_graph.as_default():
				self.sess = tf.Session(graph=detection_graph, config = config)
				rospy.loginfo('  ## Tensorflow session open: Starting inference... ##')
		except ValueError:
			rospy.logerr('   ## Error when openning session. Please restart the node ##')
			rospy.logerr(ValueError)	
		
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
		subs_topic = '/cv_camera/image_raw'
		try:
			subs_topic = rospy.get_param(rospy.get_name()+'/camera_topic')
		except:
			rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/camera_topic'))
		self._sub = rospy.Subscriber( subs_topic , Image, self.img_callback, queue_size=1, buff_size=2**24)
		
	# Define publishers
	def publishers_def(self):
		out_img_topic = '/image_objects_detect'
		try:
			out_img_topic = rospy.get_param(rospy.get_name()+'/out_img_topic')
		except:
			rospy.logwarn(' ROS was unable to load parameter '+ rospy.resolve_name(rospy.get_name()+'/out_img_topic'))
		self._img_publisher = rospy.Publisher( out_img_topic , Image, queue_size=0)
		#self._pub = rospy.Publisher('result', Int16, queue_size=1)
		
	# Camera image callback
	def img_callback(self, image_msg):
		now = rospy.get_rostime()
		# Get image as np
		image_np = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
		
		# # Actual detection
		image_np_expanded = np.expand_dims(image_np, axis=0)
		(boxes_out, scores_out, classes_out, num_detections_out) = self.sess.run(
			[self.boxes, self.scores, self.classes, self.num_detections],
			feed_dict={self.image_tensor: image_np_expanded})
		
		# Visualization of the results of a detection.
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			np.squeeze(boxes_out),
			np.squeeze(classes_out).astype(np.int32),
			np.squeeze(scores_out),
			self.category_index,
			use_normalized_coordinates=True,
			line_thickness=8)
			
		# Publish the results
		try:
			self._img_publisher.publish(self._cv_bridge.cv2_to_imgmsg(image_np, "bgr8"))
			rospy.loginfo("  Publishing inference at %s FPS", 1.0/float(rospy.Time.now().to_sec() - self.now.to_sec()))
			self.now =rospy.Time.now()
		except CvBridgeError as e:
			rospy.logerr(e)    
	
	# Spin once
	def spin(self):
		rospy.spin()
