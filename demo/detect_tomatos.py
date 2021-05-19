from PIL import Image
import PIL
import tensorflow as tf
from tensorflow.python.platform import gfile
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

tf.get_logger().setLevel('ERROR')

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

PATH_TO_SAVED_MODEL = '~/model/saved_model'
PATH_TO_LABELS = '~/model/labels.pbtxt'
NUM_CLASSES = 1

PATH_TO_TEST_IMAGES_DIR = '~/demo/images'
IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/*.png')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `1`, we know that this corresponds to `tomato`. 
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code
def load_image_into_numpy_array(image_path):
  image = Image.open(image_path)
  return np.array(image)

def main():
  print('Loading model...', end='')
  start_time = time.time()

  # Load saved model and build the detection function
  detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

  end_time = time.time()
  elapsed_time = end_time - start_time
  print('Done! Took {} seconds'.format(elapsed_time))

  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

  # Iterating over images
  for image_path in IMAGE_PATHS:
    print('Running inference for {}... '.format(image_path), end='')
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}

    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()

    vis_util.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=10,
          min_score_thresh=.10,
          agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)

    im = Image.fromarray(image_np_with_detections)
    im.save(image_path)

    print('Done')
plt.show()


main()
