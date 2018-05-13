# ROS Tensorflow objects detector
A ROS package to run any Tensorflow model for object detection or image segmentation using a camera video feed.

Tested on a Drive PX2 running a [SSD-Mobilenet](https://github.com/tensorflow/models/tree/master/research/object_detection) at 19FPS. It takes about **3mins** to open a session, so **be patient** when running this node.

## Maintainers
* David Torres: [DavidTorresOcana](mailto:david.torres.ocana@gmail.com)

## Requirements

* ROS Kinetic
* Python 2
    * cv2
    * Tensorflow for GPU
    * cv-bridge
    `$ sudo apt-get install ros-kinetic-cv-bridge ros-kinetic-cv-camera`
* CUDA and CuDNN

## Package description
### Node

Main node is `detector_node.py`

* Publish (default, user selectable): /image_objects_detect (sensor_msgs/Image)
* Subscribe (default, user selectable): /cv_camera/image_raw (sensor_msgs/Image)

### Tensorflow model
Model definitions must be placed inside `/include/ros_object_detector`

As example, the model used is a SSD-Inception detecting potholes
* [SSD-Inception and other Object detection models](https://github.com/tensorflow/models/tree/master/research/object_detection)

Models' files:
* Graph: Graph must be saved as a frozen graph .pb file in `/include/ros_potholes_detector/name-of-your-model/frozen_inference_graph.pb`
* Labels definition: .pbtxt fiel defining objects being detected. Must be placed in `/include/ros_potholes_detector/name-of-your-model/object-detection.pbtxt`

### Parameters
* ~tf_model/pb_path (string: default "/pothole_detector/pothole-graph/ssd-inception-30000-27-03-18/frozen_inference_graph.pb")  model's checkpoint .pb file path relative to /include/ros_potholes_detector
* ~tf_model/labels_definition (string: default "/pothole_detector/object-detection.pbtxt")  labels' .pbtxt file path relative to /include/ros_potholes_detector
* ~tf_model/num_objects (int: default 1) Number of objects: Number of objects defined in $labels_definition
* ~camera_topic (string: default /cv_camera/image_raw) Subscribed camera topic input to the model: ROS image topic of message type sensor_msgs.msg/Image
* ~out_img_topic (string: default /image_objects_detect) Output image topic name: message type sensor_msgs.msg/Image

### Usage
There are three options to use this package.

use a camera:
```
$ rosrun cv_camera cv_camera_node
```

#### 1. Using default model and topics
Usig default parameters and model definition:
```
$ rosrun ros_object_detector detector_node.py

```
 
#### 2. Using your model and topics: Use configuration file

If you want to configure the node to set different parameters, you can use `roslaunch` and `/config/default.yaml` file.
* Edit `/config/default.yaml` to set your parameters: Your mdoel's definition, topics, etc
* Use `roslaunch`:

```
$ roslaunch ros_object_detector ros_object_detector.launch

```
or, to launch also the camera node
```
$ roslaunch ros_object_detector ros_object_detector_all.launch

```
#### 3. Using your model and topics: on command line

You can also set any of the parameters when calling the node, by using any of the arguments below.
```
$ rosrun ros_object_detector detector_node.py _tf_model/pb_path:=path-to-your-models-pb.pb _tf_model/labels_definition:=path-to-your-labels.pbtxt _tf_model/num_objects:=your-models-num-objects _camera_topic:=input-camera-topic _out_img_topic:=output-image-topic

```
E.g.: To use a different camera topic:
```
$ rosrun ros_object_detector detector_node.py _camera_topic:=input-camera-topic

```
Note that, otherwise specified, node will use default parameters.

## TODOs
* Include other Tensorflow models like YOLOv2
* Output objects detected in a different Topic/format

