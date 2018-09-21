Model                   | Sizes                                             | Input AR  | Anchors                       | Classes               |
------------------------|---------------------------------------------------|-----------|-------------------------------|-----------------------|  
wide_yolov2_nexar_py2.h5| 720x1280 -> (416x608 -> 13x19x50) -> 720x1280     |   0.56    | yolo_wide_nexar_anchors.txt   | nexar_classes.txt     |  Nexar vehicle rears detector. Working 
yolov2_coco_py2.h5      | 424x640 -> (416x416 -> 13x13x425) -> 424x640      | 0.66-0.75 | yolo_anchors.txt              | coco_classes.txt      |  General object detector
yolov2_nexar_py2.h5     | 720x1280 -> (416x416 -> 13x13x50) -> 720x1280     |   0.56    | yolo_wide_nexar_anchors.txt   | nexar_classes.txt     |  Nexar vehicle rears detector. NOT working

                    