
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:

from threading import Thread
import numpy as np
import os
import sys
import tensorflow as tf
import operator
from os import system as kmt
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops




import cv2

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)


print("asdas",cv2.CAP_PROP_FPS)
# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model

# In[5]:

"""
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

"""
# ## Load a (frozen) Tensorflow model into memory.

# In[6]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[10]:


def run_inference_for_single_image(image, graph):
    
  with graph.as_default():
      
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

  return output_dict


# In[11]:

# KOMUTLARI YOLLAYAN FONKSİYON
def komut (detection_box,kordinat):
    if detection_box != []:
        
        for i in detection_box:
            print(i[4])

            if i[4] == "cup":
                kmt("bash -c "+'"'"/mnt/c/Users/oneorzero/Desktop/flutter/i_am_rich/lib/newItem.sh " + str(kordinat[0])+" "+str(kordinat[1])+" "+str(kordinat[2])+" "+str(kordinat[3])+'"')
   #             kmt("bash -c ""/mnt/c/Users/oneorzero/Desktop/flutter/i_am_rich/lib/appbar.sh""")
    #            kmt("bash -c ""/mnt/c/Users/oneorzero/Desktop/flutter/i_am_rich/lib/btnnavgbar.sh""")
                kmt('python C:\\Users\\oneorzero\\Desktop\\flutter\\i_am_rich\\lib\\ex1.py')
    


kordinat = [0,0,0,0]
with tf.Session(graph=detection_graph) as sess:
    asd = True   
    while True:

        ret, image_np = cap.read()
        image_np = cv2.resize(image_np,(393,623))

        #image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)



        image_np = cv2.flip(image_np,1) 
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=3)
        
        
        
        boxes = np.squeeze(output_dict['detection_boxes'])
        scores = np.squeeze(output_dict['detection_scores'])
        min_score_thresh = 0.5
        bboxes = boxes[scores > min_score_thresh]
        im_width, im_height = 393,623
        final_box = []
        img_deneme = np.zeros((623,393,3), np.uint8)
        for box in bboxes:
            ymin, xmin, ymax, xmax = box
            final_box.append([xmin * im_width, ymin * im_height,xmax * im_width, ymax * im_height])
            """
            width= 393
            height= 623
            temp_left = (width*(xmin * im_width))/500
            temp_top = ((ymin * im_height)*height)/500 
            temp_x = (width*(xmax * im_width))/500 - temp_left
            temp_y = (height*(ymax * im_height))/500 - temp_top 
            cv2.rectangle(img_deneme,(int(temp_left),int(temp_top)),(int(temp_x),int(temp_y)),(255,255,255),1)
            #cv2.imshow("kordinat", img_deneme)
          #  print("uzunluk",len(bboxes))
            """



        indexes = [k for k,v in enumerate(output_dict['detection_scores']) if (v > 0.5)]
        num_entities = len(indexes)
        try:
            class_id = operator.itemgetter(*indexes)(output_dict['detection_classes'])
            scores = operator.itemgetter(*indexes)(output_dict['detection_scores'])
        except TypeError:
            print("hic nesne yok!")
        if num_entities == 1:
            class_name = str(category_index[class_id]['name'])
            final_box[0].append(class_name)
            #print("tek",str(class_name),len(indexes))
            # if class_name == 'keyboard'and x123== True:
        else:
            for i in range(0, len(indexes)):
              #  print("girdi",len(indexes),indexes)
                final_box[i].append(category_index[class_id[i]]['name'])
              #qprint("fazla"+str(i),category_index[class_id[i]]['name'])
               # kmt(". /home/oneorzero/Desktop/tuada/lib/appbar.sh")
                
      

        kordinat = final_box
        
        if len(kordinat)==1 and kordinat[0][4] =="keyboard" :
            kordinat = final_box[0]
            kmt("bash -c "+'"'"/mnt/c/Users/oneorzero/Desktop/flutter/i_am_rich/lib/newItem.sh " + str(kordinat[0])+" "+str(kordinat[1])+" "+str(kordinat[2])+" "+str(kordinat[3])+'"')
            kmt('python C:\\Users\\oneorzero\\Desktop\\flutter\\i_am_rich\\lib\\ex1.py')     
        cv2.imshow('Video', image_np)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            


"""
     boxes = np.squeeze(output_dict['detection_boxes'])
      scores = np.squeeze(output_dict['detection_scores'])
      #set a min thresh score, say 0.8
      min_score_thresh = 0.2
      bboxes = boxes[scores > min_score_thresh]
      im_width, im_height = 500,500
      final_box = []
      print(bboxes)
      categori_listesi = []
      
      indexes = [k for k,v in enumerate(output_dict['detection_scores']) if (v > 0.01)]
      num_entities = len(indexes)
#Extract the class id
      class_id = operator.itemgetter(*indexes)(output_dict['detection_classes'])
      scores = operator.itemgetter(*indexes)(output_dict['detection_scores'])

#Convert the class id in their name
      class_names = []
      if num_entities == 1:
          class_names.append(category_index[class_id]['name'])
          class_name = str(class_names)
      else:
          for i in range(0, len(indexes)):
              categori_listesi.append(category_index[class_id[i]]['name'])
              for box in bboxes:
                  ymin, xmin, ymax, xmax = box
                  final_box.append([xmin * im_width, ymin * im_height,xmax * im_width, ymax * im_height,categori_listesi])
"""






"""
      boxes = np.squeeze(output_dict['detection_boxes'])
      scores = np.squeeze(output_dict['detection_scores'])
      #set a min thresh score, say 0.8
      min_score_thresh = 0.2
      bboxes = boxes[scores > min_score_thresh]
      im_width, im_height = 500,500
      final_box = []
      for box in bboxes:
          ymin, xmin, ymax, xmax = box
  
          print ('Top left')
          print (xmin,ymin,)
          print ('Bottom right')
          print (xmax,ymax)
        
          final_box.append([xmin * im_width, ymin * im_height,xmax * im_width, ymax * im_height])
          

      indexes = [k for k,v in enumerate(output_dict['detection_scores']) if (v > 0.01)]
      num_entities = len(indexes)
#Extract the class id
      class_id = operator.itemgetter(*indexes)(output_dict['detection_classes'])
      scores = operator.itemgetter(*indexes)(output_dict['detection_scores'])

#Convert the class id in their name
      class_names = []
      if num_entities == 1:
          class_names.append(category_index[class_id]['name'])
          class_name = str(class_names)
      else:
          for i in range(0, len(indexes)):
              print(category_index[class_id[i]]['name'])
#Number of entities

      cv2.imshow('Video', image_np)
      if cv2.waitKey(50) & 0xFF == ord('q'):
          
          cv2.destroyAllWindows()
          break
"""









""" SON
      boxes = np.squeeze(output_dict['detection_boxes'])
      scores = np.squeeze(output_dict['detection_scores'])
      #set a min thresh score, say 0.8
      min_score_thresh = 0.01
      bboxes = boxes[scores > min_score_thresh]
      im_width, im_height = 500,500
      
      categori_listesi = []
      
      indexes = [k for k,v in enumerate(output_dict['detection_scores']) if (v > 0.01)]
      num_entities = len(indexes)
#Extract the class id
      class_id = operator.itemgetter(*indexes)(output_dict['detection_classes'])
      scores = operator.itemgetter(*indexes)(output_dict['detection_scores'])

#Convert the class id in their name
      class_names = []
      class_names.append(category_index[class_id]['name'])
      print(len(bboxes))
      for box in range(len(bboxes)):
          ymin, xmin, ymax, xmax = bboxes[box]
          if num_entities == 1:
              
              class_name = str(class_names)
              final_box.append([xmin * im_width, ymin * im_height,xmax * im_width, ymax * im_height,class_name]) 
          else:
              
              final_box.append([xmin * im_width, ymin * im_height,xmax * im_width, ymax * im_height,category_index[class_id[box]]['name']])  

"""















































     
