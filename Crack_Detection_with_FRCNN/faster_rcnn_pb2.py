#import argparse
#import matplotlib.pyplot as plt
from PIL import Image
from numpy import *
'''
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model_path", help="model file path")
parser.add_argument("-i", "--image", help="Image file path")

args = parser.parse_args()
PATH_TO_FROZEN_GRAPH = args.model_path
PATH_TO_LABELS = ""
image_path = args.image
'''
PATH_TO_FROZEN_GRAPH = r"frozen_inference_graph.pb"
#image_path = r"C:\Users\1592361\Music\013-48.jpg"

import tensorflow as tf
print(tf.__version__)
import os
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
print(os.getcwd())
#import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
IMAGE_SIZE = (12, 8)
PATH_TO_LABELS = ""
# ==============================================================================
#import ops
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A module for helper tensorflow ops."""
import collections
import math
import numpy as np
import six

import tensorflow as tf
###########################################standard_fields###########################################################
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains classes specifying naming conventions used for object detection.


Specifies:
  InputDataFields: standard fields used by reader/preprocessor/batcher.
  DetectionResultFields: standard fields returned by object detector.
  BoxListFields: standard field used by BoxList
  TfExampleFields: standard fields for tf-example data format (go/tf-example).
"""


class InputDataFields(object):
  """Names for the input tensors.

  Holds the standard data field names to use for identifying input tensors. This
  should be used by the decoder to identify keys for the returned tensor_dict
  containing input tensors. And it should be used by the model to identify the
  tensors it needs.

  Attributes:
    image: image.
    image_additional_channels: additional channels.
    original_image: image in the original input size.
    original_image_spatial_shape: image in the original input size.
    key: unique key corresponding to image.
    source_id: source of the original image.
    filename: original filename of the dataset (without common path).
    groundtruth_image_classes: image-level class labels.
    groundtruth_image_confidences: image-level class confidences.
    groundtruth_boxes: coordinates of the ground truth boxes in the image.
    groundtruth_classes: box-level class labels.
    groundtruth_confidences: box-level class confidences. The shape should be
      the same as the shape of groundtruth_classes.
    groundtruth_label_types: box-level label types (e.g. explicit negative).
    groundtruth_is_crowd: [DEPRECATED, use groundtruth_group_of instead]
      is the groundtruth a single object or a crowd.
    groundtruth_area: area of a groundtruth segment.
    groundtruth_difficult: is a `difficult` object
    groundtruth_group_of: is a `group_of` objects, e.g. multiple objects of the
      same class, forming a connected group, where instances are heavily
      occluding each other.
    proposal_boxes: coordinates of object proposal boxes.
    proposal_objectness: objectness score of each proposal.
    groundtruth_instance_masks: ground truth instance masks.
    groundtruth_instance_boundaries: ground truth instance boundaries.
    groundtruth_instance_classes: instance mask-level class labels.
    groundtruth_keypoints: ground truth keypoints.
    groundtruth_keypoint_visibilities: ground truth keypoint visibilities.
    groundtruth_label_weights: groundtruth label weights.
    groundtruth_weights: groundtruth weight factor for bounding boxes.
    num_groundtruth_boxes: number of groundtruth boxes.
    is_annotated: whether an image has been labeled or not.
    true_image_shapes: true shapes of images in the resized images, as resized
      images can be padded with zeros.
    multiclass_scores: the label score per class for each box.
  """
  image = 'image'
  image_additional_channels = 'image_additional_channels'
  original_image = 'original_image'
  original_image_spatial_shape = 'original_image_spatial_shape'
  key = 'key'
  source_id = 'source_id'
  filename = 'filename'
  groundtruth_image_classes = 'groundtruth_image_classes'
  groundtruth_image_confidences = 'groundtruth_image_confidences'
  groundtruth_boxes = 'groundtruth_boxes'
  groundtruth_classes = 'groundtruth_classes'
  groundtruth_confidences = 'groundtruth_confidences'
  groundtruth_label_types = 'groundtruth_label_types'
  groundtruth_is_crowd = 'groundtruth_is_crowd'
  groundtruth_area = 'groundtruth_area'
  groundtruth_difficult = 'groundtruth_difficult'
  groundtruth_group_of = 'groundtruth_group_of'
  proposal_boxes = 'proposal_boxes'
  proposal_objectness = 'proposal_objectness'
  groundtruth_instance_masks = 'groundtruth_instance_masks'
  groundtruth_instance_boundaries = 'groundtruth_instance_boundaries'
  groundtruth_instance_classes = 'groundtruth_instance_classes'
  groundtruth_keypoints = 'groundtruth_keypoints'
  groundtruth_keypoint_visibilities = 'groundtruth_keypoint_visibilities'
  groundtruth_label_weights = 'groundtruth_label_weights'
  groundtruth_weights = 'groundtruth_weights'
  num_groundtruth_boxes = 'num_groundtruth_boxes'
  is_annotated = 'is_annotated'
  true_image_shape = 'true_image_shape'
  multiclass_scores = 'multiclass_scores'


class DetectionResultFields(object):
  """Naming conventions for storing the output of the detector.

  Attributes:
    source_id: source of the original image.
    key: unique key corresponding to image.
    detection_boxes: coordinates of the detection boxes in the image.
    detection_scores: detection scores for the detection boxes in the image.
    detection_classes: detection-level class labels.
    detection_masks: contains a segmentation mask for each detection box.
    detection_boundaries: contains an object boundary for each detection box.
    detection_keypoints: contains detection keypoints for each detection box.
    num_detections: number of detections in the batch.
  """

  source_id = 'source_id'
  key = 'key'
  detection_boxes = 'detection_boxes'
  detection_scores = 'detection_scores'
  detection_classes = 'detection_classes'
  detection_masks = 'detection_masks'
  detection_boundaries = 'detection_boundaries'
  detection_keypoints = 'detection_keypoints'
  num_detections = 'num_detections'


class BoxListFields(object):
  """Naming conventions for BoxLists.

  Attributes:
    boxes: bounding box coordinates.
    classes: classes per bounding box.
    scores: scores per bounding box.
    weights: sample weights per bounding box.
    objectness: objectness score per bounding box.
    masks: masks per bounding box.
    boundaries: boundaries per bounding box.
    keypoints: keypoints per bounding box.
    keypoint_heatmaps: keypoint heatmaps per bounding box.
    is_crowd: is_crowd annotation per bounding box.
  """
  boxes = 'boxes'
  classes = 'classes'
  scores = 'scores'
  weights = 'weights'
  confidences = 'confidences'
  objectness = 'objectness'
  masks = 'masks'
  boundaries = 'boundaries'
  keypoints = 'keypoints'
  keypoint_heatmaps = 'keypoint_heatmaps'
  is_crowd = 'is_crowd'


class TfExampleFields(object):
  """TF-example proto feature names for object detection.

  Holds the standard feature names to load from an Example proto for object
  detection.

  Attributes:
    image_encoded: JPEG encoded string
    image_format: image format, e.g. "JPEG"
    filename: filename
    channels: number of channels of image
    colorspace: colorspace, e.g. "RGB"
    height: height of image in pixels, e.g. 462
    width: width of image in pixels, e.g. 581
    source_id: original source of the image
    image_class_text: image-level label in text format
    image_class_label: image-level label in numerical format
    object_class_text: labels in text format, e.g. ["person", "cat"]
    object_class_label: labels in numbers, e.g. [16, 8]
    object_bbox_xmin: xmin coordinates of groundtruth box, e.g. 10, 30
    object_bbox_xmax: xmax coordinates of groundtruth box, e.g. 50, 40
    object_bbox_ymin: ymin coordinates of groundtruth box, e.g. 40, 50
    object_bbox_ymax: ymax coordinates of groundtruth box, e.g. 80, 70
    object_view: viewpoint of object, e.g. ["frontal", "left"]
    object_truncated: is object truncated, e.g. [true, false]
    object_occluded: is object occluded, e.g. [true, false]
    object_difficult: is object difficult, e.g. [true, false]
    object_group_of: is object a single object or a group of objects
    object_depiction: is object a depiction
    object_is_crowd: [DEPRECATED, use object_group_of instead]
      is the object a single object or a crowd
    object_segment_area: the area of the segment.
    object_weight: a weight factor for the object's bounding box.
    instance_masks: instance segmentation masks.
    instance_boundaries: instance boundaries.
    instance_classes: Classes for each instance segmentation mask.
    detection_class_label: class label in numbers.
    detection_bbox_ymin: ymin coordinates of a detection box.
    detection_bbox_xmin: xmin coordinates of a detection box.
    detection_bbox_ymax: ymax coordinates of a detection box.
    detection_bbox_xmax: xmax coordinates of a detection box.
    detection_score: detection score for the class label and box.
  """
  image_encoded = 'image/encoded'
  image_format = 'image/format'  # format is reserved keyword
  filename = 'image/filename'
  channels = 'image/channels'
  colorspace = 'image/colorspace'
  height = 'image/height'
  width = 'image/width'
  source_id = 'image/source_id'
  image_class_text = 'image/class/text'
  image_class_label = 'image/class/label'
  object_class_text = 'image/object/class/text'
  object_class_label = 'image/object/class/label'
  object_bbox_ymin = 'image/object/bbox/ymin'
  object_bbox_xmin = 'image/object/bbox/xmin'
  object_bbox_ymax = 'image/object/bbox/ymax'
  object_bbox_xmax = 'image/object/bbox/xmax'
  object_view = 'image/object/view'
  object_truncated = 'image/object/truncated'
  object_occluded = 'image/object/occluded'
  object_difficult = 'image/object/difficult'
  object_group_of = 'image/object/group_of'
  object_depiction = 'image/object/depiction'
  object_is_crowd = 'image/object/is_crowd'
  object_segment_area = 'image/object/segment/area'
  object_weight = 'image/object/weight'
  instance_masks = 'image/segmentation/object'
  instance_boundaries = 'image/boundaries/object'
  instance_classes = 'image/segmentation/object/class'
  detection_class_label = 'image/detection/label'
  detection_bbox_ymin = 'image/detection/bbox/ymin'
  detection_bbox_xmin = 'image/detection/bbox/xmin'
  detection_bbox_ymax = 'image/detection/bbox/ymax'
  detection_bbox_xmax = 'image/detection/bbox/xmax'
  detection_score = 'image/detection/score'
#########################################################################################################################
#import standard_fields as fields
#import shape_utils
#import static_shape
######################################################static_shape#######################################################
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helper functions to access TensorShape values.

The rank 4 tensor_shape must be of the form [batch_size, height, width, depth].
"""


def get_batch_size(tensor_shape):
  """Returns batch size from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the batch size of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[0].value


def get_height(tensor_shape):
  """Returns height from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the height of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[1].value


def get_width(tensor_shape):
  """Returns width from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the width of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[2].value


def get_depth(tensor_shape):
  """Returns depth from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the depth of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[3].value
########################################################################################################################
######################################################shape_utils#######################################################

import tensorflow as tf

#import static_shape


def _is_tensor(t):
  """Returns a boolean indicating whether the input is a tensor.

  Args:
    t: the input to be tested.

  Returns:
    a boolean that indicates whether t is a tensor.
  """
  return isinstance(t, (tf.Tensor, tf.SparseTensor, tf.Variable))


def _set_dim_0(t, d0):
  """Sets the 0-th dimension of the input tensor.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    d0: an integer indicating the 0-th dimension of the input tensor.

  Returns:
    the tensor t with the 0-th dimension set.
  """
  t_shape = t.get_shape().as_list()
  t_shape[0] = d0
  t.set_shape(t_shape)
  return t


def pad_tensor(t, length):
  """Pads the input tensor with 0s along the first dimension up to the length.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after padding, assuming length <= t.shape[0].

  Returns:
    padded_t: the padded tensor, whose first dimension is length. If the length
      is an integer, the first dimension of padded_t is set to length
      statically.
  """
  t_rank = tf.rank(t)
  t_shape = tf.shape(t)
  t_d0 = t_shape[0]
  pad_d0 = tf.expand_dims(length - t_d0, 0)
  pad_shape = tf.cond(
      tf.greater(t_rank, 1), lambda: tf.concat([pad_d0, t_shape[1:]], 0),
      lambda: tf.expand_dims(length - t_d0, 0))
  padded_t = tf.concat([t, tf.zeros(pad_shape, dtype=t.dtype)], 0)
  if not _is_tensor(length):
    padded_t = _set_dim_0(padded_t, length)
  return padded_t


def clip_tensor(t, length):
  """Clips the input tensor along the first dimension up to the length.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after clipping, assuming length <= t.shape[0].

  Returns:
    clipped_t: the clipped tensor, whose first dimension is length. If the
      length is an integer, the first dimension of clipped_t is set to length
      statically.
  """
  clipped_t = tf.gather(t, tf.range(length))
  if not _is_tensor(length):
    clipped_t = _set_dim_0(clipped_t, length)
  return clipped_t


def pad_or_clip_tensor(t, length):
  """Pad or clip the input tensor along the first dimension.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after processing.

  Returns:
    processed_t: the processed tensor, whose first dimension is length. If the
      length is an integer, the first dimension of the processed tensor is set
      to length statically.
  """
  return pad_or_clip_nd(t, [length] + t.shape.as_list()[1:])


def pad_or_clip_nd(tensor, output_shape):
  """Pad or Clip given tensor to the output shape.

  Args:
    tensor: Input tensor to pad or clip.
    output_shape: A list of integers / scalar tensors (or None for dynamic dim)
      representing the size to pad or clip each dimension of the input tensor.

  Returns:
    Input tensor padded and clipped to the output shape.
  """
  tensor_shape = tf.shape(tensor)
  clip_size = [
      tf.where(tensor_shape[i] - shape > 0, shape, -1)
      if shape is not None else -1 for i, shape in enumerate(output_shape)
  ]
  clipped_tensor = tf.slice(
      tensor,
      begin=tf.zeros(len(clip_size), dtype=tf.int32),
      size=clip_size)

  # Pad tensor if the shape of clipped tensor is smaller than the expected
  # shape.
  clipped_tensor_shape = tf.shape(clipped_tensor)
  trailing_paddings = [
      shape - clipped_tensor_shape[i] if shape is not None else 0
      for i, shape in enumerate(output_shape)
  ]
  paddings = tf.stack(
      [
          tf.zeros(len(trailing_paddings), dtype=tf.int32),
          trailing_paddings
      ],
      axis=1)
  padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
  output_static_shape = [
      dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
  ]
  padded_tensor.set_shape(output_static_shape)
  return padded_tensor


def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def static_or_dynamic_map_fn(fn, elems, dtype=None,
                             parallel_iterations=32, back_prop=True):
  """Runs map_fn as a (static) for loop when possible.

  This function rewrites the map_fn as an explicit unstack input -> for loop
  over function calls -> stack result combination.  This allows our graphs to
  be acyclic when the batch size is static.
  For comparison, see https://www.tensorflow.org/api_docs/python/tf/map_fn.

  Note that `static_or_dynamic_map_fn` currently is not *fully* interchangeable
  with the default tf.map_fn function as it does not accept nested inputs (only
  Tensors or lists of Tensors).  Likewise, the output of `fn` can only be a
  Tensor or list of Tensors.

  TODO(jonathanhuang): make this function fully interchangeable with tf.map_fn.

  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same structure as elems. Its output must have the
      same structure as elems.
    elems: A tensor or list of tensors, each of which will
      be unpacked along their first dimension. The sequence of the
      resulting slices will be applied to fn.
    dtype:  (optional) The output type(s) of fn. If fn returns a structure of
      Tensors differing from the structure of elems, then dtype is not optional
      and must have the same structure as the output of fn.
    parallel_iterations: (optional) number of batch items to process in
      parallel.  This flag is only used if the native tf.map_fn is used
      and defaults to 32 instead of 10 (unlike the standard tf.map_fn default).
    back_prop: (optional) True enables support for back propagation.
      This flag is only used if the native tf.map_fn is used.

  Returns:
    A tensor or sequence of tensors. Each tensor packs the
    results of applying fn to tensors unpacked from elems along the first
    dimension, from first to last.
  Raises:
    ValueError: if `elems` a Tensor or a list of Tensors.
    ValueError: if `fn` does not return a Tensor or list of Tensors
  """
  if isinstance(elems, list):
    for elem in elems:
      if not isinstance(elem, tf.Tensor):
        raise ValueError('`elems` must be a Tensor or list of Tensors.')

    elem_shapes = [elem.shape.as_list() for elem in elems]
    # Fall back on tf.map_fn if shapes of each entry of `elems` are None or fail
    # to all be the same size along the batch dimension.
    for elem_shape in elem_shapes:
      if (not elem_shape or not elem_shape[0]
          or elem_shape[0] != elem_shapes[0][0]):
        return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    arg_tuples = zip(*[tf.unstack(elem) for elem in elems])
    outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
  else:
    if not isinstance(elems, tf.Tensor):
      raise ValueError('`elems` must be a Tensor or list of Tensors.')
    elems_shape = elems.shape.as_list()
    if not elems_shape or not elems_shape[0]:
      return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    outputs = [fn(arg) for arg in tf.unstack(elems)]
  # Stack `outputs`, which is a list of Tensors or list of lists of Tensors
  if all([isinstance(output, tf.Tensor) for output in outputs]):
    return tf.stack(outputs)
  else:
    if all([isinstance(output, list) for output in outputs]):
      if all([all(
          [isinstance(entry, tf.Tensor) for entry in output_list])
              for output_list in outputs]):
        return [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
  raise ValueError('`fn` should return a Tensor or a list of Tensors.')


def check_min_image_dim(min_dim, image_tensor):
  """Checks that the image width/height are greater than some number.

  This function is used to check that the width and height of an image are above
  a certain value. If the image shape is static, this function will perform the
  check at graph construction time. Otherwise, if the image shape varies, an
  Assertion control dependency will be added to the graph.

  Args:
    min_dim: The minimum number of pixels along the width and height of the
             image.
    image_tensor: The image tensor to check size for.

  Returns:
    If `image_tensor` has dynamic size, return `image_tensor` with a Assert
    control dependency. Otherwise returns image_tensor.

  Raises:
    ValueError: if `image_tensor`'s' width or height is smaller than `min_dim`.
  """
  image_shape = image_tensor.get_shape()
  image_height = get_height(image_shape)
  image_width = get_width(image_shape)
  if image_height is None or image_width is None:
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(image_tensor)[1], min_dim),
                       tf.greater_equal(tf.shape(image_tensor)[2], min_dim)),
        ['image size must be >= {} in both height and width.'.format(min_dim)])
    with tf.control_dependencies([shape_assert]):
      return tf.identity(image_tensor)

  if image_height < min_dim or image_width < min_dim:
    raise ValueError(
        'image size must be >= %d in both height and width; image dim = %d,%d' %
        (min_dim, image_height, image_width))

  return image_tensor


def assert_shape_equal(shape_a, shape_b):
  """Asserts that shape_a and shape_b are equal.

  If the shapes are static, raises a ValueError when the shapes
  mismatch.

  If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
  mismatch.

  Args:
    shape_a: a list containing shape of the first tensor.
    shape_b: a list containing shape of the second tensor.

  Returns:
    Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
    when the shapes are dynamic.

  Raises:
    ValueError: When shapes are both static and unequal.
  """
  if (all(isinstance(dim, int) for dim in shape_a) and
      all(isinstance(dim, int) for dim in shape_b)):
    if shape_a != shape_b:
      raise ValueError('Unequal shapes {}, {}'.format(shape_a, shape_b))
    else: return tf.no_op()
  else:
    return tf.assert_equal(shape_a, shape_b)


def assert_shape_equal_along_first_dimension(shape_a, shape_b):
  """Asserts that shape_a and shape_b are the same along the 0th-dimension.

  If the shapes are static, raises a ValueError when the shapes
  mismatch.

  If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
  mismatch.

  Args:
    shape_a: a list containing shape of the first tensor.
    shape_b: a list containing shape of the second tensor.

  Returns:
    Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
    when the shapes are dynamic.

  Raises:
    ValueError: When shapes are both static and unequal.
  """
  if isinstance(shape_a[0], int) and isinstance(shape_b[0], int):
    if shape_a[0] != shape_b[0]:
      raise ValueError('Unequal first dimension {}, {}'.format(
          shape_a[0], shape_b[0]))
    else: return tf.no_op()
  else:
    return tf.assert_equal(shape_a[0], shape_b[0])


def assert_box_normalized(boxes, maximum_normalized_coordinate=1.1):
  """Asserts the input box tensor is normalized.

  Args:
    boxes: a tensor of shape [N, 4] where N is the number of boxes.
    maximum_normalized_coordinate: Maximum coordinate value to be considered
      as normalized, default to 1.1.

  Returns:
    a tf.Assert op which fails when the input box tensor is not normalized.

  Raises:
    ValueError: When the input box tensor is not normalized.
  """
  box_minimum = tf.reduce_min(boxes)
  box_maximum = tf.reduce_max(boxes)
  return tf.Assert(
      tf.logical_and(
          tf.less_equal(box_maximum, maximum_normalized_coordinate),
          tf.greater_equal(box_minimum, 0)),
      [boxes])
      
##########################################################################################################
def expanded_shape(orig_shape, start_dim, num_dims):
  """Inserts multiple ones into a shape vector.

  Inserts an all-1 vector of length num_dims at position start_dim into a shape.
  Can be combined with tf.reshape to generalize tf.expand_dims.

  Args:
    orig_shape: the shape into which the all-1 vector is added (int32 vector)
    start_dim: insertion position (int scalar)
    num_dims: length of the inserted all-1 vector (int scalar)
  Returns:
    An int32 vector of length tf.size(orig_shape) + num_dims.
  """
  with tf.name_scope('ExpandedShape'):
    start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
    before = tf.slice(orig_shape, [0], start_dim)
    add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
    after = tf.slice(orig_shape, start_dim, [-1])
    new_shape = tf.concat([before, add_shape, after], 0)
    return new_shape


def normalized_to_image_coordinates(normalized_boxes, image_shape,
                                    parallel_iterations=32):
  """Converts a batch of boxes from normal to image coordinates.

  Args:
    normalized_boxes: a float32 tensor of shape [None, num_boxes, 4] in
      normalized coordinates.
    image_shape: a float32 tensor of shape [4] containing the image shape.
    parallel_iterations: parallelism for the map_fn op.

  Returns:
    absolute_boxes: a float32 tensor of shape [None, num_boxes, 4] containing
      the boxes in image coordinates.
  """
  x_scale = tf.cast(image_shape[2], tf.float32)
  y_scale = tf.cast(image_shape[1], tf.float32)
  def _to_absolute_coordinates(normalized_boxes):
    y_min, x_min, y_max, x_max = tf.split(
        value=normalized_boxes, num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxes = tf.concat([y_min, x_min, y_max, x_max], 1)
    return scaled_boxes

  absolute_boxes = static_or_dynamic_map_fn(
      _to_absolute_coordinates,
      elems=(normalized_boxes),
      dtype=tf.float32,
      parallel_iterations=parallel_iterations,
      back_prop=True)
  return absolute_boxes


def meshgrid(x, y):
  """Tiles the contents of x and y into a pair of grids.

  Multidimensional analog of numpy.meshgrid, giving the same behavior if x and y
  are vectors. Generally, this will give:

  xgrid(i1, ..., i_m, j_1, ..., j_n) = x(j_1, ..., j_n)
  ygrid(i1, ..., i_m, j_1, ..., j_n) = y(i_1, ..., i_m)

  Keep in mind that the order of the arguments and outputs is reverse relative
  to the order of the indices they go into, done for compatibility with numpy.
  The output tensors have the same shapes.  Specifically:

  xgrid.get_shape() = y.get_shape().concatenate(x.get_shape())
  ygrid.get_shape() = y.get_shape().concatenate(x.get_shape())

  Args:
    x: A tensor of arbitrary shape and rank. xgrid will contain these values
       varying in its last dimensions.
    y: A tensor of arbitrary shape and rank. ygrid will contain these values
       varying in its first dimensions.
  Returns:
    A tuple of tensors (xgrid, ygrid).
  """
  with tf.name_scope('Meshgrid'):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
    y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))

    xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
    new_shape = y.get_shape().concatenate(x.get_shape())
    xgrid.set_shape(new_shape)
    ygrid.set_shape(new_shape)

    return xgrid, ygrid


def fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def pad_to_multiple(tensor, multiple):
  """Returns the tensor zero padded to the specified multiple.

  Appends 0s to the end of the first and second dimension (height and width) of
  the tensor until both dimensions are a multiple of the input argument
  'multiple'. E.g. given an input tensor of shape [1, 3, 5, 1] and an input
  multiple of 4, PadToMultiple will append 0s so that the resulting tensor will
  be of shape [1, 4, 8, 1].

  Args:
    tensor: rank 4 float32 tensor, where
            tensor -> [batch_size, height, width, channels].
    multiple: the multiple to pad to.

  Returns:
    padded_tensor: the tensor zero padded to the specified multiple.
  """
  if multiple == 1:
    return tensor

  tensor_shape = tensor.get_shape()
  batch_size = get_batch_size(tensor_shape)
  tensor_height = get_height(tensor_shape)
  tensor_width = get_width(tensor_shape)
  tensor_depth = get_depth(tensor_shape)

  if batch_size is None:
    batch_size = tf.shape(tensor)[0]

  if tensor_height is None:
    tensor_height = tf.shape(tensor)[1]
    padded_tensor_height = tf.to_int32(
        tf.ceil(tf.to_float(tensor_height) / tf.to_float(multiple))) * multiple
  else:
    padded_tensor_height = int(
        math.ceil(float(tensor_height) / multiple) * multiple)

  if tensor_width is None:
    tensor_width = tf.shape(tensor)[2]
    padded_tensor_width = tf.to_int32(
        tf.ceil(tf.to_float(tensor_width) / tf.to_float(multiple))) * multiple
  else:
    padded_tensor_width = int(
        math.ceil(float(tensor_width) / multiple) * multiple)

  if tensor_depth is None:
    tensor_depth = tf.shape(tensor)[3]

  # Use tf.concat instead of tf.pad to preserve static shape
  if padded_tensor_height != tensor_height:
    height_pad = tf.zeros([
        batch_size, padded_tensor_height - tensor_height, tensor_width,
        tensor_depth
    ])
    tensor = tf.concat([tensor, height_pad], 1)
  if padded_tensor_width != tensor_width:
    width_pad = tf.zeros([
        batch_size, padded_tensor_height, padded_tensor_width - tensor_width,
        tensor_depth
    ])
    tensor = tf.concat([tensor, width_pad], 2)

  return tensor


def padded_one_hot_encoding(indices, depth, left_pad):
  """Returns a zero padded one-hot tensor.

  This function converts a sparse representation of indices (e.g., [4]) to a
  zero padded one-hot representation (e.g., [0, 0, 0, 0, 1] with depth = 4 and
  left_pad = 1). If `indices` is empty, the result will simply be a tensor of
  shape (0, depth + left_pad). If depth = 0, then this function just returns
  `None`.

  Args:
    indices: an integer tensor of shape [num_indices].
    depth: depth for the one-hot tensor (integer).
    left_pad: number of zeros to left pad the one-hot tensor with (integer).

  Returns:
    padded_onehot: a tensor with shape (num_indices, depth + left_pad). Returns
      `None` if the depth is zero.

  Raises:
    ValueError: if `indices` does not have rank 1 or if `left_pad` or `depth are
      either negative or non-integers.

  TODO(rathodv): add runtime checks for depth and indices.
  """
  if depth < 0 or not isinstance(depth, six.integer_types):
    raise ValueError('`depth` must be a non-negative integer.')
  if left_pad < 0 or not isinstance(left_pad, six.integer_types):
    raise ValueError('`left_pad` must be a non-negative integer.')
  if depth == 0:
    return None

  rank = len(indices.get_shape().as_list())
  if rank != 1:
    raise ValueError('`indices` must have rank 1, but has rank=%s' % rank)

  def one_hot_and_pad():
    one_hot = tf.cast(tf.one_hot(tf.cast(indices, tf.int64), depth,
                                 on_value=1, off_value=0), tf.float32)
    return tf.pad(one_hot, [[0, 0], [left_pad, 0]], mode='CONSTANT')
  result = tf.cond(tf.greater(tf.size(indices), 0), one_hot_and_pad,
                   lambda: tf.zeros((depth + left_pad, 0)))
  return tf.reshape(result, [-1, depth + left_pad])


def dense_to_sparse_boxes(dense_locations, dense_num_boxes, num_classes):
  """Converts bounding boxes from dense to sparse form.

  Args:
    dense_locations:  a [max_num_boxes, 4] tensor in which only the first k rows
      are valid bounding box location coordinates, where k is the sum of
      elements in dense_num_boxes.
    dense_num_boxes: a [max_num_classes] tensor indicating the counts of
       various bounding box classes e.g. [1, 0, 0, 2] means that the first
       bounding box is of class 0 and the second and third bounding boxes are
       of class 3. The sum of elements in this tensor is the number of valid
       bounding boxes.
    num_classes: number of classes

  Returns:
    box_locations: a [num_boxes, 4] tensor containing only valid bounding
       boxes (i.e. the first num_boxes rows of dense_locations)
    box_classes: a [num_boxes] tensor containing the classes of each bounding
       box (e.g. dense_num_boxes = [1, 0, 0, 2] => box_classes = [0, 3, 3]
  """

  num_valid_boxes = tf.reduce_sum(dense_num_boxes)
  box_locations = tf.slice(dense_locations,
                           tf.constant([0, 0]), tf.stack([num_valid_boxes, 4]))
  tiled_classes = [tf.tile([i], tf.expand_dims(dense_num_boxes[i], 0))
                   for i in range(num_classes)]
  box_classes = tf.concat(tiled_classes, 0)
  box_locations.set_shape([None, 4])
  return box_locations, box_classes


def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=tf.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.

  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

  Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.

  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  """
  size = tf.to_int32(size)
  zeros = tf.ones([size], dtype=dtype) * default_value
  values = tf.ones_like(indices, dtype=dtype) * indices_value

  return tf.dynamic_stitch([tf.range(size), tf.to_int32(indices)],
                           [zeros, values])


def reduce_sum_trailing_dimensions(tensor, ndims):
  """Computes sum across all dimensions following first `ndims` dimensions."""
  return tf.reduce_sum(tensor, axis=tuple(range(ndims, tensor.shape.ndims)))


def retain_groundtruth(tensor_dict, valid_indices):
  input_shape = valid_indices.get_shape().as_list()
  if not (len(input_shape) == 1 or
          (len(input_shape) == 2 and input_shape[1] == 1)):
    raise ValueError('The shape of valid_indices is invalid.')
  valid_indices = tf.reshape(valid_indices, [-1])
  valid_dict = {}
  if InputDataFields.groundtruth_boxes in tensor_dict:
    # Prevents reshape failure when num_boxes is 0.
    num_boxes = tf.maximum(tf.shape(
        tensor_dict[InputDataFields.groundtruth_boxes])[0], 1)
    for key in tensor_dict:
      if key in [InputDataFields.groundtruth_boxes,
                 InputDataFields.groundtruth_classes,
                 InputDataFields.groundtruth_keypoints,
                 InputDataFields.groundtruth_instance_masks]:
        valid_dict[key] = tf.gather(tensor_dict[key], valid_indices)
      # Input decoder returns empty tensor when these fields are not provided.
      # Needs to reshape into [num_boxes, -1] for tf.gather() to work.
      elif key in [InputDataFields.groundtruth_is_crowd,
                   InputDataFields.groundtruth_area,
                   InputDataFields.groundtruth_difficult,
                   InputDataFields.groundtruth_label_types]:
        valid_dict[key] = tf.reshape(
            tf.gather(tf.reshape(tensor_dict[key], [num_boxes, -1]),
                      valid_indices), [-1])
      # Fields that are not associated with boxes.
      else:
        valid_dict[key] = tensor_dict[key]
  else:
    raise ValueError('%s not present in input tensor dict.' % (
        InputDataFields.groundtruth_boxes))
  return valid_dict


def retain_groundtruth_with_positive_classes(tensor_dict):
  """Retains only groundtruth with positive class ids.

  Args:
    tensor_dict: a dictionary of following groundtruth tensors -
      
  Returns:
    a dictionary of tensors containing only the groundtruth with positive
    classes.

  Raises:
    ValueError: If groundtruth_classes tensor is not in tensor_dict.
  """
  if InputDataFields.groundtruth_classes not in tensor_dict:
    raise ValueError('`groundtruth classes` not in tensor_dict.')
  keep_indices = tf.where(tf.greater(
      tensor_dict[InputDataFields.groundtruth_classes], 0))
  return retain_groundtruth(tensor_dict, keep_indices)


def replace_nan_groundtruth_label_scores_with_ones(label_scores):
  """Replaces nan label scores with 1.0.

  Args:
    label_scores: a tensor containing object annoation label scores.

  Returns:
    a tensor where NaN label scores have been replaced by ones.
  """
  return tf.where(
      tf.is_nan(label_scores), tf.ones(tf.shape(label_scores)), label_scores)


def filter_groundtruth_with_crowd_boxes(tensor_dict):
  """Filters out groundtruth with boxes corresponding to crowd.

  Args:
    tensor_dict: a dictionary of following groundtruth tensors -
      
  Returns:
    a dictionary of tensors containing only the groundtruth that have bounding
    boxes.
  """
  if InputDataFields.groundtruth_is_crowd in tensor_dict:
    is_crowd = tensor_dict[InputDataFields.groundtruth_is_crowd]
    is_not_crowd = tf.logical_not(is_crowd)
    is_not_crowd_indices = tf.where(is_not_crowd)
    tensor_dict = retain_groundtruth(tensor_dict, is_not_crowd_indices)
  return tensor_dict


def filter_groundtruth_with_nan_box_coordinates(tensor_dict):
  """Filters out groundtruth with no bounding boxes.

  Args:
    tensor_dict: a dictionary of following groundtruth tensors -
     
  Returns:
    a dictionary of tensors containing only the groundtruth that have bounding
    boxes.
  """
  groundtruth_boxes = tensor_dict[InputDataFields.groundtruth_boxes]
  nan_indicator_vector = tf.greater(tf.reduce_sum(tf.to_int32(
      tf.is_nan(groundtruth_boxes)), reduction_indices=[1]), 0)
  valid_indicator_vector = tf.logical_not(nan_indicator_vector)
  valid_indices = tf.where(valid_indicator_vector)

  return retain_groundtruth(tensor_dict, valid_indices)


def normalize_to_target(inputs,
                        target_norm_value,
                        dim,
                        epsilon=1e-7,
                        trainable=True,
                        scope='NormalizeToTarget',
                        summarize=True):
  """L2 normalizes the inputs across the specified dimension to a target norm.

  This op implements the L2 Normalization layer introduced in
  Liu, Wei, et al. "SSD: Single Shot MultiBox Detector."
  and Liu, Wei, Andrew Rabinovich, and Alexander C. Berg.
  "Parsenet: Looking wider to see better." and is useful for bringing
  activations from multiple layers in a convnet to a standard scale.

  Note that the rank of `inputs` must be known and the dimension to which
  normalization is to be applied should be statically defined.

  TODO(jonathanhuang): Add option to scale by L2 norm of the entire input.

  Args:
    inputs: A `Tensor` of arbitrary size.
    target_norm_value: A float value that specifies an initial target norm or
      a list of floats (whose length must be equal to the depth along the
      dimension to be normalized) specifying a per-dimension multiplier
      after normalization.
    dim: The dimension along which the input is normalized.
    epsilon: A small value to add to the inputs to avoid dividing by zero.
    trainable: Whether the norm is trainable or not
    scope: Optional scope for variable_scope.
    summarize: Whether or not to add a tensorflow summary for the op.

  Returns:
    The input tensor normalized to the specified target norm.

  Raises:
    ValueError: If dim is smaller than the number of dimensions in 'inputs'.
    ValueError: If target_norm_value is not a float or a list of floats with
      length equal to the depth along the dimension to be normalized.
  """
  with tf.variable_scope(scope, 'NormalizeToTarget', [inputs]):
    if not inputs.get_shape():
      raise ValueError('The input rank must be known.')
    input_shape = inputs.get_shape().as_list()
    input_rank = len(input_shape)
    if dim < 0 or dim >= input_rank:
      raise ValueError(
          'dim must be non-negative but smaller than the input rank.')
    if not input_shape[dim]:
      raise ValueError('input shape should be statically defined along '
                       'the specified dimension.')
    depth = input_shape[dim]
    if not (isinstance(target_norm_value, float) or
            (isinstance(target_norm_value, list) and
             len(target_norm_value) == depth) and
            all([isinstance(val, float) for val in target_norm_value])):
      raise ValueError('target_norm_value must be a float or a list of floats '
                       'with length equal to the depth along the dimension to '
                       'be normalized.')
    if isinstance(target_norm_value, float):
      initial_norm = depth * [target_norm_value]
    else:
      initial_norm = target_norm_value
    target_norm = tf.contrib.framework.model_variable(
        name='weights', dtype=tf.float32,
        initializer=tf.constant(initial_norm, dtype=tf.float32),
        trainable=trainable)
    if summarize:
      mean = tf.reduce_mean(target_norm)
      mean = tf.Print(mean, ['NormalizeToTarget:', mean])
      tf.summary.scalar(tf.get_variable_scope().name, mean)
    lengths = epsilon + tf.sqrt(tf.reduce_sum(tf.square(inputs), dim, True))
    mult_shape = input_rank*[1]
    mult_shape[dim] = depth
    return tf.reshape(target_norm, mult_shape) * tf.truediv(inputs, lengths)


def batch_position_sensitive_crop_regions(images,
                                          boxes,
                                          crop_size,
                                          num_spatial_bins,
                                          global_pool,
                                          parallel_iterations=64):
  """Position sensitive crop with batches of images and boxes.

  This op is exactly like `position_sensitive_crop_regions` below but operates
  on batches of images and boxes. See `position_sensitive_crop_regions` function
  below for the operation applied per batch element.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 3-D tensor of shape `[batch, num_boxes, 4]`. Each box is specified in
      normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value
      of `y` is mapped to the image coordinate at `y * (image_height - 1)`, so
      as the `[0, 1]` interval of normalized image height is mapped to
      `[0, image_height - 1] in image height coordinates. We do allow y1 > y2,
      in which case the sampled crop is an up-down flipped version of the
      original image. The width dimension is treated similarly.
    crop_size: See `position_sensitive_crop_regions` below.
    num_spatial_bins: See `position_sensitive_crop_regions` below.
    global_pool: See `position_sensitive_crop_regions` below.
    parallel_iterations: Number of batch items to process in parallel.

  Returns:
  """
  def _position_sensitive_crop_fn(inputs):
    images, boxes = inputs
    return position_sensitive_crop_regions(
        images,
        boxes,
        crop_size=crop_size,
        num_spatial_bins=num_spatial_bins,
        global_pool=global_pool)

  return static_or_dynamic_map_fn(
      _position_sensitive_crop_fn,
      elems=[images, boxes],
      dtype=tf.float32,
      parallel_iterations=parallel_iterations)


def position_sensitive_crop_regions(image,
                                    boxes,
                                    crop_size,
                                    num_spatial_bins,
                                    global_pool):
  """Position-sensitive crop and pool rectangular regions from a feature grid.

  The output crops are split into `spatial_bins_y` vertical bins
  and `spatial_bins_x` horizontal bins. For each intersection of a vertical
  and a horizontal bin the output values are gathered by performing
  `tf.image.crop_and_resize` (bilinear resampling) on a a separate subset of
  channels of the image. This reduces `depth` by a factor of
  `(spatial_bins_y * spatial_bins_x)`.

  When global_pool is True, this function implements a differentiable version
  of position-sensitive RoI pooling used in
  [R-FCN detection system](https://arxiv.org/abs/1605.06409).

  When global_pool is False, this function implements a differentiable version
  of position-sensitive assembling operation used in
  [instance FCN](https://arxiv.org/abs/1603.08678).

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
      A 3-D tensor of shape `[image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 2-D tensor of shape `[num_boxes, 4]`. Each box is specified in
      normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value
      of `y` is mapped to the image coordinate at `y * (image_height - 1)`, so
      as the `[0, 1]` interval of normalized image height is mapped to
      `[0, image_height - 1] in image height coordinates. We do allow y1 > y2,
      in which case the sampled crop is an up-down flipped version of the
      original image. The width dimension is treated similarly.
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    num_spatial_bins: A list of two integers `[spatial_bins_y, spatial_bins_x]`.
      Represents the number of position-sensitive bins in y and x directions.
      Both values should be >= 1. `crop_height` should be divisible by
      `spatial_bins_y`, and similarly for width.
      The number of image channels should be divisible by
      (spatial_bins_y * spatial_bins_x).
      Suggested value from R-FCN paper: [3, 3].
    global_pool: A boolean variable.
      If True, we perform average global pooling on the features assembled from
        the position-sensitive score maps.
      If False, we keep the position-pooled features without global pooling
        over the spatial coordinates.
      Note that using global_pool=True is equivalent to but more efficient than
        running the function with global_pool=False and then performing global
        average pooling.

  Returns:
    position_sensitive_features: A 4-D tensor of shape
      `[num_boxes, K, K, crop_channels]`,
      where `crop_channels = depth / (spatial_bins_y * spatial_bins_x)`,
      where K = 1 when global_pool is True (Average-pooled cropped regions),
      and K = crop_size when global_pool is False.
  Raises:
    ValueError: Raised in four situations:
      `num_spatial_bins` is not >= 1;
      `num_spatial_bins` does not divide `crop_size`;
      `(spatial_bins_y*spatial_bins_x)` does not divide `depth`;
      `bin_crop_size` is not square when global_pool=False due to the
        constraint in function space_to_depth.
  """
  total_bins = 1
  bin_crop_size = []

  for (num_bins, crop_dim) in zip(num_spatial_bins, crop_size):
    if num_bins < 1:
      raise ValueError('num_spatial_bins should be >= 1')

    if crop_dim % num_bins != 0:
      raise ValueError('crop_size should be divisible by num_spatial_bins')

    total_bins *= num_bins
    bin_crop_size.append(crop_dim // num_bins)

  if not global_pool and bin_crop_size[0] != bin_crop_size[1]:
    raise ValueError('Only support square bin crop size for now.')

  ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
  spatial_bins_y, spatial_bins_x = num_spatial_bins

  # Split each box into spatial_bins_y * spatial_bins_x bins.
  position_sensitive_boxes = []
  for bin_y in range(spatial_bins_y):
    step_y = (ymax - ymin) / spatial_bins_y
    for bin_x in range(spatial_bins_x):
      step_x = (xmax - xmin) / spatial_bins_x
      box_coordinates = [ymin + bin_y * step_y,
                         xmin + bin_x * step_x,
                         ymin + (bin_y + 1) * step_y,
                         xmin + (bin_x + 1) * step_x,
                        ]
      position_sensitive_boxes.append(tf.stack(box_coordinates, axis=1))

  image_splits = tf.split(value=image, num_or_size_splits=total_bins, axis=2)

  image_crops = []
  for (split, box) in zip(image_splits, position_sensitive_boxes):
    if split.shape.is_fully_defined() and box.shape.is_fully_defined():
      crop = tf.squeeze(
          matmul_crop_and_resize(
              tf.expand_dims(split, axis=0), tf.expand_dims(box, axis=0),
              bin_crop_size),
          axis=0)
    else:
      crop = tf.image.crop_and_resize(
          tf.expand_dims(split, 0), box,
          tf.zeros(tf.shape(boxes)[0], dtype=tf.int32), bin_crop_size)
    image_crops.append(crop)

  if global_pool:
    # Average over all bins.
    position_sensitive_features = tf.add_n(image_crops) / len(image_crops)
    # Then average over spatial positions within the bins.
    position_sensitive_features = tf.reduce_mean(
        position_sensitive_features, [1, 2], keep_dims=True)
  else:
    # Reorder height/width to depth channel.
    block_size = bin_crop_size[0]
    if block_size >= 2:
      image_crops = [tf.space_to_depth(
          crop, block_size=block_size) for crop in image_crops]

    # Pack image_crops so that first dimension is for position-senstive boxes.
    position_sensitive_features = tf.stack(image_crops, axis=0)

    # Unroll the position-sensitive boxes to spatial positions.
    position_sensitive_features = tf.squeeze(
        tf.batch_to_space_nd(position_sensitive_features,
                             block_shape=[1] + num_spatial_bins,
                             crops=tf.zeros((3, 2), dtype=tf.int32)),
        squeeze_dims=[0])

    # Reorder back the depth channel.
    if block_size >= 2:
      position_sensitive_features = tf.depth_to_space(
          position_sensitive_features, block_size=block_size)

  return position_sensitive_features


def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
  """Transforms the box masks back to full image masks.

  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.

  Args:
    box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.

  Returns:
    A tf.float32 tensor of size [num_masks, image_height, image_width].
  """
  # TODO(rathodv): Make this a public function.
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    return tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reverse_boxes,
        box_ind=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        extrapolation_value=0.0)
  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
  return tf.squeeze(image_masks, axis=3)


def merge_boxes_with_multiple_labels(boxes,
                                     classes,
                                     confidences,
                                     num_classes,
                                     quantization_bins=10000):
  """Merges boxes with same coordinates and returns K-hot encoded classes.

  Args:
    boxes: A tf.float32 tensor with shape [N, 4] holding N boxes. Only
      normalized coordinates are allowed.
    classes: A tf.int32 tensor with shape [N] holding class indices.
      The class index starts at 0.
    confidences: A tf.float32 tensor with shape [N] holding class confidences.
    num_classes: total number of classes to use for K-hot encoding.
    quantization_bins: the number of bins used to quantize the box coordinate.

  Returns:
    merged_boxes: A tf.float32 tensor with shape [N', 4] holding boxes,
      where N' <= N.
    class_encodings: A tf.int32 tensor with shape [N', num_classes] holding
      K-hot encodings for the merged boxes.
    confidence_encodings: A tf.float32 tensor with shape [N', num_classes]
      holding encodings of confidences for the merged boxes.
    merged_box_indices: A tf.int32 tensor with shape [N'] holding original
      indices of the boxes.
  """
  boxes_shape = tf.shape(boxes)
  classes_shape = tf.shape(classes)
  confidences_shape = tf.shape(confidences)
  box_class_shape_assert = assert_shape_equal_along_first_dimension(
      boxes_shape, classes_shape)
  box_confidence_shape_assert = (
      assert_shape_equal_along_first_dimension(
          boxes_shape, confidences_shape))
  box_dimension_assert = tf.assert_equal(boxes_shape[1], 4)
  box_normalized_assert = assert_box_normalized(boxes)

  with tf.control_dependencies(
      [box_class_shape_assert, box_confidence_shape_assert,
       box_dimension_assert, box_normalized_assert]):
    quantized_boxes = tf.to_int64(boxes * (quantization_bins - 1))
    ymin, xmin, ymax, xmax = tf.unstack(quantized_boxes, axis=1)
    hashcodes = (
        ymin +
        xmin * quantization_bins +
        ymax * quantization_bins * quantization_bins +
        xmax * quantization_bins * quantization_bins * quantization_bins)
    unique_hashcodes, unique_indices = tf.unique(hashcodes)
    num_boxes = tf.shape(boxes)[0]
    num_unique_boxes = tf.shape(unique_hashcodes)[0]
    merged_box_indices = tf.unsorted_segment_min(
        tf.range(num_boxes), unique_indices, num_unique_boxes)
    merged_boxes = tf.gather(boxes, merged_box_indices)

    def map_box_encodings(i):
      """Produces box K-hot and score encodings for each class index."""
      box_mask = tf.equal(
          unique_indices, i * tf.ones(num_boxes, dtype=tf.int32))
      box_mask = tf.reshape(box_mask, [-1])
      box_indices = tf.boolean_mask(classes, box_mask)
      box_confidences = tf.boolean_mask(confidences, box_mask)
      box_class_encodings = tf.sparse_to_dense(
          box_indices, [num_classes], 1, validate_indices=False)
      box_confidence_encodings = tf.sparse_to_dense(
          box_indices, [num_classes], box_confidences, validate_indices=False)
      return box_class_encodings, box_confidence_encodings

    class_encodings, confidence_encodings = tf.map_fn(
        map_box_encodings,
        tf.range(num_unique_boxes),
        back_prop=False,
        dtype=(tf.int32, tf.float32))

    merged_boxes = tf.reshape(merged_boxes, [-1, 4])
    class_encodings = tf.reshape(class_encodings, [-1, num_classes])
    confidence_encodings = tf.reshape(confidence_encodings, [-1, num_classes])
    merged_box_indices = tf.reshape(merged_box_indices, [-1])
    return (merged_boxes, class_encodings, confidence_encodings,
            merged_box_indices)


def nearest_neighbor_upsampling(input_tensor, scale=None, height_scale=None,
                                width_scale=None):
  """Nearest neighbor upsampling implementation.

  Nearest neighbor upsampling function that maps input tensor with shape
  [batch_size, height, width, channels] to [batch_size, height * scale
  , width * scale, channels]. This implementation only uses reshape and
  broadcasting to make it TPU compatible.

  Args:
    input_tensor: A float32 tensor of size [batch, height_in, width_in,
      channels].
    scale: An integer multiple to scale resolution of input data in both height
      and width dimensions.
    height_scale: An integer multiple to scale the height of input image. This
      option when provided overrides `scale` option.
    width_scale: An integer multiple to scale the width of input image. This
      option when provided overrides `scale` option.
  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].

  Raises:
    ValueError: If both scale and height_scale or if both scale and width_scale
      are None.
  """
  if not scale and (height_scale is None or width_scale is None):
    raise ValueError('Provide either `scale` or `height_scale` and'
                     ' `width_scale`.')
  with tf.name_scope('nearest_neighbor_upsampling'):
    h_scale = scale if height_scale is None else height_scale
    w_scale = scale if width_scale is None else width_scale
    (batch_size, height, width,
     channels) = combined_static_and_dynamic_shape(input_tensor)
    output_tensor = tf.reshape(
        input_tensor, [batch_size, height, 1, width, 1, channels]) * tf.ones(
            [1, 1, h_scale, 1, w_scale, 1], dtype=input_tensor.dtype)
    return tf.reshape(output_tensor,
                      [batch_size, height * h_scale, width * w_scale, channels])


def matmul_gather_on_zeroth_axis(params, indices, scope=None):
  """Matrix multiplication based implementation of tf.gather on zeroth axis.

  TODO(rathodv, jonathanhuang): enable sparse matmul option.

  Args:
    params: A float32 Tensor. The tensor from which to gather values.
      Must be at least rank 1.
    indices: A Tensor. Must be one of the following types: int32, int64.
      Must be in range [0, params.shape[0])
    scope: A name for the operation (optional).

  Returns:
    A Tensor. Has the same type as params. Values from params gathered
    from indices given by indices, with shape indices.shape + params.shape[1:].
  """
  with tf.name_scope(scope, 'MatMulGather'):
    params_shape = combined_static_and_dynamic_shape(params)
    indices_shape = combined_static_and_dynamic_shape(indices)
    params2d = tf.reshape(params, [params_shape[0], -1])
    indicator_matrix = tf.one_hot(indices, params_shape[0])
    gathered_result_flattened = tf.matmul(indicator_matrix, params2d)
    return tf.reshape(gathered_result_flattened,
                      tf.stack(indices_shape + params_shape[1:]))


def matmul_crop_and_resize(image, boxes, crop_size, scope=None):
  """Matrix multiplication based implementation of the crop and resize op.

  Extracts crops from the input image tensor and bilinearly resizes them
  (possibly with aspect ratio change) to a common output size specified by
  crop_size. This is more general than the crop_to_bounding_box op which
  extracts a fixed size slice from the input image and does not allow
  resizing or aspect ratio change.

  Returns a tensor with crops from the input image at positions defined at
  the bounding box locations in boxes. The cropped boxes are all resized
  (with bilinear interpolation) to a fixed size = `[crop_height, crop_width]`.
  The result is a 5-D tensor `[batch, num_boxes, crop_height, crop_width,
  depth]`.

  Running time complexity:
    O((# channels) * (# boxes) * (crop_size)^2 * M), where M is the number
  of pixels of the longer edge of the image.

  Note that this operation is meant to replicate the behavior of the standard
  tf.image.crop_and_resize operation but there are a few differences.
  Specifically:
    1) The extrapolation value (the values that are interpolated from outside
      the bounds of the image window) is always zero
    2) Only XLA supported operations are used (e.g., matrix multiplication).
    3) There is no `box_indices` argument --- to run this op on multiple images,
      one must currently call this op independently on each image.
    4) All shapes and the `crop_size` parameter are assumed to be statically
      defined.  Moreover, the number of boxes must be strictly nonzero.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`, `half`, 'bfloat16', `float32`, `float64`.
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32` or 'bfloat16'.
      A 3-D tensor of shape `[batch, num_boxes, 4]`. The boxes are specified in
      normalized coordinates and are of the form `[y1, x1, y2, x2]`. A
      normalized coordinate value of `y` is mapped to the image coordinate at
      `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1] in image height coordinates.
      We do allow y1 > y2, in which case the sampled crop is an up-down flipped
      version of the original image. The width dimension is treated similarly.
      Normalized coordinates outside the `[0, 1]` range are allowed, in which
      case we use `extrapolation_value` to extrapolate the input image values.
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    scope: A name for the operation (optional).

  Returns:
    A 5-D tensor of shape `[batch, num_boxes, crop_height, crop_width, depth]`

  Raises:
    ValueError: if image tensor does not have shape
      `[batch, image_height, image_width, depth]` and all dimensions statically
      defined.
    ValueError: if boxes tensor does not have shape `[batch, num_boxes, 4]`
      where num_boxes > 0.
    ValueError: if crop_size is not a list of two positive integers
  """
  img_shape = image.shape.as_list()
  boxes_shape = boxes.shape.as_list()
  _, img_height, img_width, _ = img_shape
  if not isinstance(crop_size, list) or len(crop_size) != 2:
    raise ValueError('`crop_size` must be a list of length 2')
  dimensions = img_shape + crop_size + boxes_shape
  if not all([isinstance(dim, int) for dim in dimensions]):
    raise ValueError('all input shapes must be statically defined')
  if len(boxes_shape) != 3 or boxes_shape[2] != 4:
    raise ValueError('`boxes` should have shape `[batch, num_boxes, 4]`')
  if len(img_shape) != 4:
    raise ValueError('image should have shape '
                     '`[batch, image_height, image_width, depth]`')
  num_crops = boxes_shape[0]
  if not num_crops > 0:
    raise ValueError('number of boxes must be > 0')
  if not (crop_size[0] > 0 and crop_size[1] > 0):
    raise ValueError('`crop_size` must be a list of two positive integers.')

  def _lin_space_weights(num, img_size):
    if num > 1:
      start_weights = tf.linspace(img_size - 1.0, 0.0, num)
      stop_weights = img_size - 1 - start_weights
    else:
      start_weights = tf.constant(num * [.5 * (img_size - 1)], dtype=tf.float32)
      stop_weights = tf.constant(num * [.5 * (img_size - 1)], dtype=tf.float32)
    return (start_weights, stop_weights)

  with tf.name_scope(scope, 'MatMulCropAndResize'):
    y1_weights, y2_weights = _lin_space_weights(crop_size[0], img_height)
    x1_weights, x2_weights = _lin_space_weights(crop_size[1], img_width)
    y1_weights = tf.cast(y1_weights, boxes.dtype)
    y2_weights = tf.cast(y2_weights, boxes.dtype)
    x1_weights = tf.cast(x1_weights, boxes.dtype)
    x2_weights = tf.cast(x2_weights, boxes.dtype)
    [y1, x1, y2, x2] = tf.unstack(boxes, axis=2)

    # Pixel centers of input image and grid points along height and width
    image_idx_h = tf.constant(
        np.reshape(np.arange(img_height), (1, 1, 1, img_height)),
        dtype=boxes.dtype)
    image_idx_w = tf.constant(
        np.reshape(np.arange(img_width), (1, 1, 1, img_width)),
        dtype=boxes.dtype)
    grid_pos_h = tf.expand_dims(
        tf.einsum('ab,c->abc', y1, y1_weights) + tf.einsum(
            'ab,c->abc', y2, y2_weights),
        axis=3)
    grid_pos_w = tf.expand_dims(
        tf.einsum('ab,c->abc', x1, x1_weights) + tf.einsum(
            'ab,c->abc', x2, x2_weights),
        axis=3)

    # Create kernel matrices of pairwise kernel evaluations between pixel
    # centers of image and grid points.
    kernel_h = tf.nn.relu(1 - tf.abs(image_idx_h - grid_pos_h))
    kernel_w = tf.nn.relu(1 - tf.abs(image_idx_w - grid_pos_w))

    # Compute matrix multiplication between the spatial dimensions of the image
    # and height-wise kernel using einsum.
    intermediate_image = tf.einsum('abci,aiop->abcop', kernel_h, image)
    # Compute matrix multiplication between the spatial dimensions of the
    # intermediate_image and width-wise kernel using einsum.
    return tf.einsum('abno,abcop->abcnp', kernel_w, intermediate_image)


def native_crop_and_resize(image, boxes, crop_size, scope=None):
  """Same as `matmul_crop_and_resize` but uses tf.image.crop_and_resize."""
  def get_box_inds(proposals):
    proposals_shape = proposals.get_shape().as_list()
    if any(dim is None for dim in proposals_shape):
      proposals_shape = tf.shape(proposals)
    ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
    multiplier = tf.expand_dims(
        tf.range(start=0, limit=proposals_shape[0]), 1)
    return tf.reshape(ones_mat * multiplier, [-1])

  with tf.name_scope(scope, 'CropAndResize'):
    cropped_regions = tf.image.crop_and_resize(
        image, tf.reshape(boxes, [-1] + boxes.shape.as_list()[2:]),
        get_box_inds(boxes), crop_size)
    final_shape = tf.concat([tf.shape(boxes)[:2],
                             tf.shape(cropped_regions)[1:]], axis=0)
    return tf.reshape(cropped_regions, final_shape)





EqualizationLossConfig = collections.namedtuple('EqualizationLossConfig',
                                                ['weight', 'exclude_prefixes'])



# ==============================================================================
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Label map utility functions."""

import logging

import tensorflow as tf
from google.protobuf import text_format
#import string_int_label_map_pb2
##############################################string_int_label_map_pb2###############################################

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/string_int_label_map.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/string_int_label_map.proto',
  package='object_detection.protos',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n2object_detection/protos/string_int_label_map.proto\x12\x17object_detection.protos\"G\n\x15StringIntLabelMapItem\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\x05\x12\x14\n\x0c\x64isplay_name\x18\x03 \x01(\t\"Q\n\x11StringIntLabelMap\x12<\n\x04item\x18\x01 \x03(\x0b\x32..object_detection.protos.StringIntLabelMapItem'
)




_STRINGINTLABELMAPITEM = _descriptor.Descriptor(
  name='StringIntLabelMapItem',
  full_name='object_detection.protos.StringIntLabelMapItem',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='object_detection.protos.StringIntLabelMapItem.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='id', full_name='object_detection.protos.StringIntLabelMapItem.id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='display_name', full_name='object_detection.protos.StringIntLabelMapItem.display_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=79,
  serialized_end=150,
)


_STRINGINTLABELMAP = _descriptor.Descriptor(
  name='StringIntLabelMap',
  full_name='object_detection.protos.StringIntLabelMap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='item', full_name='object_detection.protos.StringIntLabelMap.item', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=152,
  serialized_end=233,
)

_STRINGINTLABELMAP.fields_by_name['item'].message_type = _STRINGINTLABELMAPITEM
DESCRIPTOR.message_types_by_name['StringIntLabelMapItem'] = _STRINGINTLABELMAPITEM
DESCRIPTOR.message_types_by_name['StringIntLabelMap'] = _STRINGINTLABELMAP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

StringIntLabelMapItem = _reflection.GeneratedProtocolMessageType('StringIntLabelMapItem', (_message.Message,), {
  'DESCRIPTOR' : _STRINGINTLABELMAPITEM,
  #'__module__' : 'object_detection.protos.string_int_label_map_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.StringIntLabelMapItem)
  })
_sym_db.RegisterMessage(StringIntLabelMapItem)

StringIntLabelMap = _reflection.GeneratedProtocolMessageType('StringIntLabelMap', (_message.Message,), {
  'DESCRIPTOR' : _STRINGINTLABELMAP,
  #'__module__' : 'object_detection.protos.string_int_label_map_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.StringIntLabelMap)
  })
_sym_db.RegisterMessage(StringIntLabelMap)


# @@protoc_insertion_point(module_scope)
###########################################################################################################################

def _validate_label_map(label_map):
  """Checks if a label map is valid.

  Args:
    label_map: StringIntLabelMap to validate.

  Raises:
    ValueError: if label map is invalid.
  """
  for item in label_map.item:
    if item.id < 0:
      raise ValueError('Label map ids should be >= 0.')
    if (item.id == 0 and item.name != 'background' and
        item.display_name != 'background'):
      raise ValueError('Label map id 0 is reserved for the background label')


def create_category_index(categories):
  """Creates dictionary of COCO compatible categories keyed by category id.

  Args:
    categories: a list of dicts, each of which has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.

  Returns:
    category_index: a dict containing the same entries as categories, but keyed
      by the 'id' field of each category.
  """
  category_index = {}
  for cat in categories:
    category_index[cat['id']] = cat
  return category_index


def get_max_label_map_index(label_map):
  """Get maximum index in label map.

  Args:
    label_map: a StringIntLabelMapProto

  Returns:
    an integer
  """
  return max([item.id for item in label_map.item])


def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
  """Given label map proto returns categories list compatible with eval.

  This function converts label map proto and returns a list of dicts, each of
  which  has the following keys:
    'id': (required) an integer id uniquely identifying this category.
    'name': (required) string representing category name
      e.g., 'cat', 'dog', 'pizza'.
  We only allow class into the list if its id-label_id_offset is
  between 0 (inclusive) and max_num_classes (exclusive).
  If there are several items mapping to the same id in the label map,
  we will only keep the first one in the categories list.

  Args:
    label_map: a StringIntLabelMapProto or None.  If None, a default categories
      list is created with max_num_classes categories.
    max_num_classes: maximum number of (consecutive) label indices to include.
    use_display_name: (boolean) choose whether to load 'display_name' field as
      category name.  If False or if the display_name field does not exist, uses
      'name' field as category names instead.

  Returns:
    categories: a list of dictionaries representing all possible categories.
  """
  categories = []
  list_of_ids_already_added = []
  if not label_map:
    label_id_offset = 1
    for class_id in range(max_num_classes):
      categories.append({
          'id': class_id + label_id_offset,
          'name': 'category_{}'.format(class_id + label_id_offset)
      })
    return categories
  for item in label_map.item:
    if not 0 < item.id <= max_num_classes:
      logging.info(
          'Ignore item %d since it falls outside of requested '
          'label range.', item.id)
      continue
    if use_display_name and item.HasField('display_name'):
      name = item.display_name
    else:
      name = item.name
    if item.id not in list_of_ids_already_added:
      list_of_ids_already_added.append(item.id)
      categories.append({'id': item.id, 'name': name})
  return categories



def load_labelmap(path):
  """Loads label map proto.

  Args:
    path: path to StringIntLabelMap proto text file.
  Returns:
    a StringIntLabelMapProto
  """
  #with tf.compat.v2.io.gfile.GFile(path, 'r') as fid:
    #label_map_string = fid.read()
  label_map_string = "item {id: 1,name: 'crack'}"
  print("+++++++++++++++")
  print(label_map_string)
  print(type(label_map_string))
  print("+++++++++++++++")
  label_map = StringIntLabelMap()
  try:
    text_format.Merge(label_map_string, label_map)
  except text_format.ParseError:
    label_map.ParseFromString(label_map_string)
  _validate_label_map(label_map)
  return label_map

def get_label_map_dict(label_map_path,
                       use_display_name=False,
                       fill_in_gaps_and_background=False):
  """Reads a label map and returns a dictionary of label names to id.

  Args:
    label_map_path: path to StringIntLabelMap proto text file.
    use_display_name: whether to use the label map items' display names as keys.
    fill_in_gaps_and_background: whether to fill in gaps and background with
    respect to the id field in the proto. The id: 0 is reserved for the
    'background' class and will be added if it is missing. All other missing
    ids in range(1, max(id)) will be added with a dummy class name
    ("class_<id>") if they are missing.

  Returns:
    A dictionary mapping label names to id.

  Raises:
    ValueError: if fill_in_gaps_and_background and label_map has non-integer or
    negative values.
  """
  label_map = load_labelmap(label_map_path)
  label_map_dict = {}
  for item in label_map.item:
    if use_display_name:
      label_map_dict[item.display_name] = item.id
    else:
      label_map_dict[item.name] = item.id

  if fill_in_gaps_and_background:
    values = set(label_map_dict.values())

    if 0 not in values:
      label_map_dict['background'] = 0
    if not all(isinstance(value, int) for value in values):
      raise ValueError('The values in label map must be integers in order to'
                       'fill_in_gaps_and_background.')
    if not all(value >= 0 for value in values):
      raise ValueError('The values in the label map must be positive.')

    if len(values) != max(values) + 1:
      # there are gaps in the labels, fill in gaps.
      for value in range(1, max(values)):
        if value not in values:
          label_map_dict['class_' + str(value)] = value

  return label_map_dict


def create_categories_from_labelmap(label_map_path, use_display_name=True):
  """Reads a label map and returns categories list compatible with eval.

  This function converts label map proto and returns a list of dicts, each of
  which  has the following keys:
    'id': an integer id uniquely identifying this category.
    'name': string representing category name e.g., 'cat', 'dog'.

  Args:
    label_map_path: Path to `StringIntLabelMap` proto text file.
    use_display_name: (boolean) choose whether to load 'display_name' field
      as category name.  If False or if the display_name field does not exist,
      uses 'name' field as category names instead.

  Returns:
    categories: a list of dictionaries representing all possible categories.
  """
  label_map = load_labelmap(label_map_path)
  max_num_classes = max(item.id for item in label_map.item)
  return convert_label_map_to_categories(label_map, max_num_classes,
                                         use_display_name)


def create_category_index_from_labelmap(label_map_path, use_display_name=True):
  """Reads a label map and returns a category index.

  Args:
    label_map_path: Path to `StringIntLabelMap` proto text file.
    use_display_name: (boolean) choose whether to load 'display_name' field
      as category name.  If False or if the display_name field does not exist,
      uses 'name' field as category names instead.

  Returns:
    A category index, which is a dictionary that maps integer ids to dicts
    containing categories, e.g.
    {1: {'id': 1, 'name': 'dog'}, 2: {'id': 2, 'name': 'cat'}, ...}
  """
  categories = create_categories_from_labelmap(label_map_path, use_display_name)
  return create_category_index(categories)


def create_class_agnostic_category_index():
  """Creates a category index with a single `object` class."""
  return {1: {'id': 1, 'name': 'object'}}




# ==============================================================================
#import visualization_utils as vis_util
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
import abc
import collections
import functools
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import tensorflow as tf

#import standard_fields as fields
#from object_detection.utils import shape_utils
##################################################################################################################
import tensorflow as tf

#import static_shape


def _is_tensor(t):
  """Returns a boolean indicating whether the input is a tensor.

  Args:
    t: the input to be tested.

  Returns:
    a boolean that indicates whether t is a tensor.
  """
  return isinstance(t, (tf.Tensor, tf.SparseTensor, tf.Variable))


def _set_dim_0(t, d0):
  """Sets the 0-th dimension of the input tensor.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    d0: an integer indicating the 0-th dimension of the input tensor.

  Returns:
    the tensor t with the 0-th dimension set.
  """
  t_shape = t.get_shape().as_list()
  t_shape[0] = d0
  t.set_shape(t_shape)
  return t


def pad_tensor(t, length):
  """Pads the input tensor with 0s along the first dimension up to the length.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after padding, assuming length <= t.shape[0].

  Returns:
    padded_t: the padded tensor, whose first dimension is length. If the length
      is an integer, the first dimension of padded_t is set to length
      statically.
  """
  t_rank = tf.rank(t)
  t_shape = tf.shape(t)
  t_d0 = t_shape[0]
  pad_d0 = tf.expand_dims(length - t_d0, 0)
  pad_shape = tf.cond(
      tf.greater(t_rank, 1), lambda: tf.concat([pad_d0, t_shape[1:]], 0),
      lambda: tf.expand_dims(length - t_d0, 0))
  padded_t = tf.concat([t, tf.zeros(pad_shape, dtype=t.dtype)], 0)
  if not _is_tensor(length):
    padded_t = _set_dim_0(padded_t, length)
  return padded_t


def clip_tensor(t, length):
  """Clips the input tensor along the first dimension up to the length.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after clipping, assuming length <= t.shape[0].

  Returns:
    clipped_t: the clipped tensor, whose first dimension is length. If the
      length is an integer, the first dimension of clipped_t is set to length
      statically.
  """
  clipped_t = tf.gather(t, tf.range(length))
  if not _is_tensor(length):
    clipped_t = _set_dim_0(clipped_t, length)
  return clipped_t


def pad_or_clip_tensor(t, length):
  """Pad or clip the input tensor along the first dimension.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after processing.

  Returns:
    processed_t: the processed tensor, whose first dimension is length. If the
      length is an integer, the first dimension of the processed tensor is set
      to length statically.
  """
  return pad_or_clip_nd(t, [length] + t.shape.as_list()[1:])


def pad_or_clip_nd(tensor, output_shape):
  """Pad or Clip given tensor to the output shape.

  Args:
    tensor: Input tensor to pad or clip.
    output_shape: A list of integers / scalar tensors (or None for dynamic dim)
      representing the size to pad or clip each dimension of the input tensor.

  Returns:
    Input tensor padded and clipped to the output shape.
  """
  tensor_shape = tf.shape(tensor)
  clip_size = [
      tf.where(tensor_shape[i] - shape > 0, shape, -1)
      if shape is not None else -1 for i, shape in enumerate(output_shape)
  ]
  clipped_tensor = tf.slice(
      tensor,
      begin=tf.zeros(len(clip_size), dtype=tf.int32),
      size=clip_size)

  # Pad tensor if the shape of clipped tensor is smaller than the expected
  # shape.
  clipped_tensor_shape = tf.shape(clipped_tensor)
  trailing_paddings = [
      shape - clipped_tensor_shape[i] if shape is not None else 0
      for i, shape in enumerate(output_shape)
  ]
  paddings = tf.stack(
      [
          tf.zeros(len(trailing_paddings), dtype=tf.int32),
          trailing_paddings
      ],
      axis=1)
  padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
  output_static_shape = [
      dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
  ]
  padded_tensor.set_shape(output_static_shape)
  return padded_tensor


def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def static_or_dynamic_map_fn(fn, elems, dtype=None,
                             parallel_iterations=32, back_prop=True):
  """Runs map_fn as a (static) for loop when possible.

  This function rewrites the map_fn as an explicit unstack input -> for loop
  over function calls -> stack result combination.  This allows our graphs to
  be acyclic when the batch size is static.
  For comparison, see https://www.tensorflow.org/api_docs/python/tf/map_fn.

  Note that `static_or_dynamic_map_fn` currently is not *fully* interchangeable
  with the default tf.map_fn function as it does not accept nested inputs (only
  Tensors or lists of Tensors).  Likewise, the output of `fn` can only be a
  Tensor or list of Tensors.

  TODO(jonathanhuang): make this function fully interchangeable with tf.map_fn.

  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same structure as elems. Its output must have the
      same structure as elems.
    elems: A tensor or list of tensors, each of which will
      be unpacked along their first dimension. The sequence of the
      resulting slices will be applied to fn.
    dtype:  (optional) The output type(s) of fn. If fn returns a structure of
      Tensors differing from the structure of elems, then dtype is not optional
      and must have the same structure as the output of fn.
    parallel_iterations: (optional) number of batch items to process in
      parallel.  This flag is only used if the native tf.map_fn is used
      and defaults to 32 instead of 10 (unlike the standard tf.map_fn default).
    back_prop: (optional) True enables support for back propagation.
      This flag is only used if the native tf.map_fn is used.

  Returns:
    A tensor or sequence of tensors. Each tensor packs the
    results of applying fn to tensors unpacked from elems along the first
    dimension, from first to last.
  Raises:
    ValueError: if `elems` a Tensor or a list of Tensors.
    ValueError: if `fn` does not return a Tensor or list of Tensors
  """
  if isinstance(elems, list):
    for elem in elems:
      if not isinstance(elem, tf.Tensor):
        raise ValueError('`elems` must be a Tensor or list of Tensors.')

    elem_shapes = [elem.shape.as_list() for elem in elems]
    # Fall back on tf.map_fn if shapes of each entry of `elems` are None or fail
    # to all be the same size along the batch dimension.
    for elem_shape in elem_shapes:
      if (not elem_shape or not elem_shape[0]
          or elem_shape[0] != elem_shapes[0][0]):
        return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    arg_tuples = zip(*[tf.unstack(elem) for elem in elems])
    outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
  else:
    if not isinstance(elems, tf.Tensor):
      raise ValueError('`elems` must be a Tensor or list of Tensors.')
    elems_shape = elems.shape.as_list()
    if not elems_shape or not elems_shape[0]:
      return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    outputs = [fn(arg) for arg in tf.unstack(elems)]
  # Stack `outputs`, which is a list of Tensors or list of lists of Tensors
  if all([isinstance(output, tf.Tensor) for output in outputs]):
    return tf.stack(outputs)
  else:
    if all([isinstance(output, list) for output in outputs]):
      if all([all(
          [isinstance(entry, tf.Tensor) for entry in output_list])
              for output_list in outputs]):
        return [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
  raise ValueError('`fn` should return a Tensor or a list of Tensors.')


def check_min_image_dim(min_dim, image_tensor):
  """Checks that the image width/height are greater than some number.

  This function is used to check that the width and height of an image are above
  a certain value. If the image shape is static, this function will perform the
  check at graph construction time. Otherwise, if the image shape varies, an
  Assertion control dependency will be added to the graph.

  Args:
    min_dim: The minimum number of pixels along the width and height of the
             image.
    image_tensor: The image tensor to check size for.

  Returns:
    If `image_tensor` has dynamic size, return `image_tensor` with a Assert
    control dependency. Otherwise returns image_tensor.

  Raises:
    ValueError: if `image_tensor`'s' width or height is smaller than `min_dim`.
  """
  image_shape = image_tensor.get_shape()
  image_height = get_height(image_shape)
  image_width = get_width(image_shape)
  if image_height is None or image_width is None:
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(image_tensor)[1], min_dim),
                       tf.greater_equal(tf.shape(image_tensor)[2], min_dim)),
        ['image size must be >= {} in both height and width.'.format(min_dim)])
    with tf.control_dependencies([shape_assert]):
      return tf.identity(image_tensor)

  if image_height < min_dim or image_width < min_dim:
    raise ValueError(
        'image size must be >= %d in both height and width; image dim = %d,%d' %
        (min_dim, image_height, image_width))

  return image_tensor


def assert_shape_equal(shape_a, shape_b):
  """Asserts that shape_a and shape_b are equal.

  If the shapes are static, raises a ValueError when the shapes
  mismatch.

  If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
  mismatch.

  Args:
    shape_a: a list containing shape of the first tensor.
    shape_b: a list containing shape of the second tensor.

  Returns:
    Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
    when the shapes are dynamic.

  Raises:
    ValueError: When shapes are both static and unequal.
  """
  if (all(isinstance(dim, int) for dim in shape_a) and
      all(isinstance(dim, int) for dim in shape_b)):
    if shape_a != shape_b:
      raise ValueError('Unequal shapes {}, {}'.format(shape_a, shape_b))
    else: return tf.no_op()
  else:
    return tf.assert_equal(shape_a, shape_b)


def assert_shape_equal_along_first_dimension(shape_a, shape_b):
  """Asserts that shape_a and shape_b are the same along the 0th-dimension.

  If the shapes are static, raises a ValueError when the shapes
  mismatch.

  If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
  mismatch.

  Args:
    shape_a: a list containing shape of the first tensor.
    shape_b: a list containing shape of the second tensor.

  Returns:
    Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
    when the shapes are dynamic.

  Raises:
    ValueError: When shapes are both static and unequal.
  """
  if isinstance(shape_a[0], int) and isinstance(shape_b[0], int):
    if shape_a[0] != shape_b[0]:
      raise ValueError('Unequal first dimension {}, {}'.format(
          shape_a[0], shape_b[0]))
    else: return tf.no_op()
  else:
    return tf.assert_equal(shape_a[0], shape_b[0])


def assert_box_normalized(boxes, maximum_normalized_coordinate=1.1):
  """Asserts the input box tensor is normalized.

  Args:
    boxes: a tensor of shape [N, 4] where N is the number of boxes.
    maximum_normalized_coordinate: Maximum coordinate value to be considered
      as normalized, default to 1.1.

  Returns:
    a tf.Assert op which fails when the input box tensor is not normalized.

  Raises:
    ValueError: When the input box tensor is not normalized.
  """
  box_minimum = tf.reduce_min(boxes)
  box_maximum = tf.reduce_max(boxes)
  return tf.Assert(
      tf.logical_and(
          tf.less_equal(box_maximum, maximum_normalized_coordinate),
          tf.greater_equal(box_minimum, 0)),
      [boxes])
##################################################################################################################
_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
  """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  image_pil = Image.fromarray(image)
  draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                               display_str_list_list)
  np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
  """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)


def _visualize_boxes(image, boxes, classes, scores, category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image, boxes, classes, scores, category_index=category_index, **kwargs)


def _visualize_boxes_and_masks(image, boxes, classes, scores, masks,
                               category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      **kwargs)


def _visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints,
                                   category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      keypoints=keypoints,
      **kwargs)


def _visualize_boxes_and_masks_and_keypoints(
    image, boxes, classes, scores, masks, keypoints, category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      keypoints=keypoints,
      **kwargs)


def _resize_original_image(image, image_shape):
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_images(
      image,
      image_shape,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      align_corners=True)
  return tf.cast(tf.squeeze(image, 0), tf.uint8)


def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         original_image_spatial_shape=None,
                                         true_image_shape=None,
                                         instance_masks=None,
                                         keypoints=None,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2,
                                         use_normalized_coordinates=True):
  """Draws bounding boxes, masks, and keypoints on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C]. If C > 3, additional
      channels will be ignored. If C = 1, then we convert the images to RGB
      images.
    boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    classes: [N, max_detections] int tensor of detection classes. Note that
      classes are 1-indexed.
    scores: [N, max_detections] float32 tensor of detection scores.
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    original_image_spatial_shape: [N, 2] tensor containing the spatial size of
      the original image.
    true_image_shape: [N, 3] tensor containing the spatial size of unpadded
      original_image.
    instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with
      instance masks.
    keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
      with keypoints.
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    4D image tensor of type uint8, with boxes drawn on top.
  """
  # Additional channels are being ignored.
  if images.shape[3] > 3:
    images = images[:, :, :, 0:3]
  elif images.shape[3] == 1:
    images = tf.image.grayscale_to_rgb(images)
  visualization_keyword_args = {
      'use_normalized_coordinates': use_normalized_coordinates,
      'max_boxes_to_draw': max_boxes_to_draw,
      'min_score_thresh': min_score_thresh,
      'agnostic_mode': False,
      'line_thickness': 4
  }
  if true_image_shape is None:
    true_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 3])
  else:
    true_shapes = true_image_shape
  if original_image_spatial_shape is None:
    original_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 2])
  else:
    original_shapes = original_image_spatial_shape

  if instance_masks is not None and keypoints is None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_masks,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [
        true_shapes, original_shapes, images, boxes, classes, scores,
        instance_masks
    ]
  elif instance_masks is None and keypoints is not None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_keypoints,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [
        true_shapes, original_shapes, images, boxes, classes, scores, keypoints
    ]
  elif instance_masks is not None and keypoints is not None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_masks_and_keypoints,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [
        true_shapes, original_shapes, images, boxes, classes, scores,
        instance_masks, keypoints
    ]
  else:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [
        true_shapes, original_shapes, images, boxes, classes, scores
    ]

  def draw_boxes(image_and_detections):
    """Draws boxes on image."""
    true_shape = image_and_detections[0]
    original_shape = image_and_detections[1]
    if true_image_shape is not None:
      image = pad_or_clip_nd(image_and_detections[2],
                                         [true_shape[0], true_shape[1], 3])
    if original_image_spatial_shape is not None:
      image_and_detections[2] = _resize_original_image(image, original_shape)

    image_with_boxes = tf.py_func(visualize_boxes_fn, image_and_detections[2:],
                                  tf.uint8)
    return image_with_boxes

  images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
  return images


def draw_side_by_side_evaluation_image(eval_dict,
                                       category_index,
                                       max_boxes_to_draw=20,
                                       min_score_thresh=0.2,
                                       use_normalized_coordinates=True):
  """Creates a side-by-side image with detections and groundtruth.

  Bounding boxes (and instance masks, if available) are visualized on both
  subimages.

  Args:
    eval_dict: The evaluation dictionary returned by
      eval_util.result_dict_for_batched_example() or
      eval_util.result_dict_for_single_example().
    category_index: A category index (dictionary) produced from a labelmap.
    max_boxes_to_draw: The maximum number of boxes to draw for detections.
    min_score_thresh: The minimum score threshold for showing detections.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    A list of [1, H, 2 * W, C] uint8 tensor. The subimage on the left
      corresponds to detections, while the subimage on the right corresponds to
      groundtruth.
  """
  detection_fields = DetectionResultFields()
  input_data_fields = InputDataFields()

  images_with_detections_list = []

  # Add the batch dimension if the eval_dict is for single example.
  if len(eval_dict[detection_fields.detection_classes].shape) == 1:
    for key in eval_dict:
      if key != input_data_fields.original_image:
        eval_dict[key] = tf.expand_dims(eval_dict[key], 0)

  for indx in range(eval_dict[input_data_fields.original_image].shape[0]):
    instance_masks = None
    if detection_fields.detection_masks in eval_dict:
      instance_masks = tf.cast(
          tf.expand_dims(
              eval_dict[detection_fields.detection_masks][indx], axis=0),
          tf.uint8)
    keypoints = None
    if detection_fields.detection_keypoints in eval_dict:
      keypoints = tf.expand_dims(
          eval_dict[detection_fields.detection_keypoints][indx], axis=0)
    groundtruth_instance_masks = None
    if input_data_fields.groundtruth_instance_masks in eval_dict:
      groundtruth_instance_masks = tf.cast(
          tf.expand_dims(
              eval_dict[input_data_fields.groundtruth_instance_masks][indx],
              axis=0), tf.uint8)

    images_with_detections = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(
            eval_dict[input_data_fields.original_image][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_boxes][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_classes][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_scores][indx], axis=0),
        category_index,
        original_image_spatial_shape=tf.expand_dims(
            eval_dict[input_data_fields.original_image_spatial_shape][indx],
            axis=0),
        true_image_shape=tf.expand_dims(
            eval_dict[input_data_fields.true_image_shape][indx], axis=0),
        instance_masks=instance_masks,
        keypoints=keypoints,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates)
    images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(
            eval_dict[input_data_fields.original_image][indx], axis=0),
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_boxes][indx], axis=0),
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_classes][indx], axis=0),
        tf.expand_dims(
            tf.ones_like(
                eval_dict[input_data_fields.groundtruth_classes][indx],
                dtype=tf.float32),
            axis=0),
        category_index,
        original_image_spatial_shape=tf.expand_dims(
            eval_dict[input_data_fields.original_image_spatial_shape][indx],
            axis=0),
        true_image_shape=tf.expand_dims(
            eval_dict[input_data_fields.true_image_shape][indx], axis=0),
        instance_masks=groundtruth_instance_masks,
        keypoints=None,
        max_boxes_to_draw=None,
        min_score_thresh=0.0,
        use_normalized_coordinates=use_normalized_coordinates)
    images_with_detections_list.append(
        tf.concat([images_with_detections, images_with_groundtruth], axis=2))
  return images_with_detections_list


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
  """Draws keypoints on an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil, keypoints, color, radius,
                          use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
  """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color, fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color
      )
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates)

  return image


def add_cdf_image_summary(values, name):
  """Adds a tf.summary.image for a CDF plot of the values.

  Normalizes `values` such that they sum to 1, plots the cumulative distribution
  function and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    name: name for the image summary.
  """
  def cdf_plot(values):
    """Numpy function to plot CDF."""
    normalized_values = values / np.sum(values)
    sorted_values = np.sort(normalized_values)
    cumulative_values = np.cumsum(sorted_values)
    fraction_of_examples = (np.arange(cumulative_values.size, dtype=np.float32)
                            / cumulative_values.size)
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot('111')
    ax.plot(fraction_of_examples, cumulative_values)
    ax.set_ylabel('cumulative normalized values')
    ax.set_xlabel('fraction of examples')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
        1, int(height), int(width), 3)
    return image
  cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
  tf.summary.image(name, cdf_plot)


def add_hist_image_summary(values, bins, name):
  """Adds a tf.summary.image for a histogram plot of the values.

  Plots the histogram of values and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    bins: bin edges which will be directly passed to np.histogram.
    name: name for the image summary.
  """

  def hist_plot(values, bins):
    """Numpy function to plot hist."""
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot('111')
    y, x = np.histogram(values, bins=bins)
    ax.plot(x[:-1], y)
    ax.set_ylabel('count')
    ax.set_xlabel('value')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(
        fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            1, int(height), int(width), 3)
    return image
  hist_plot = tf.py_func(hist_plot, [values, bins], tf.uint8)
  tf.summary.image(name, hist_plot)


class EvalMetricOpsVisualization(object):
  """Abstract base class responsible for visualizations during evaluation.

  Currently, summary images are not run during evaluation. One way to produce
  evaluation images in Tensorboard is to provide tf.summary.image strings as
  `value_ops` in tf.estimator.EstimatorSpec's `eval_metric_ops`. This class is
  responsible for accruing images (with overlaid detections and groundtruth)
  and returning a dictionary that can be passed to `eval_metric_ops`.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               category_index,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='evaluation_image'):
    """Creates an EvalMetricOpsVisualization.

    Args:
      category_index: A category index (dictionary) produced from a labelmap.
      max_examples_to_draw: The maximum number of example summaries to produce.
      max_boxes_to_draw: The maximum number of boxes to draw for detections.
      min_score_thresh: The minimum score threshold for showing detections.
      use_normalized_coordinates: Whether to assume boxes and kepoints are in
        normalized coordinates (as opposed to absolute coordiantes).
        Default is True.
      summary_name_prefix: A string prefix for each image summary.
    """

    self._category_index = category_index
    self._max_examples_to_draw = max_examples_to_draw
    self._max_boxes_to_draw = max_boxes_to_draw
    self._min_score_thresh = min_score_thresh
    self._use_normalized_coordinates = use_normalized_coordinates
    self._summary_name_prefix = summary_name_prefix
    self._images = []

  def clear(self):
    self._images = []

  def add_images(self, images):
    """Store a list of images, each with shape [1, H, W, C]."""
    if len(self._images) >= self._max_examples_to_draw:
      return

    # Store images and clip list if necessary.
    self._images.extend(images)
    if len(self._images) > self._max_examples_to_draw:
      self._images[self._max_examples_to_draw:] = []

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns metric ops for use in tf.estimator.EstimatorSpec.

    Args:
      eval_dict: A dictionary that holds an image, groundtruth, and detections
        for a batched example. Note that, we use only the first example for
        visualization. See eval_util.result_dict_for_batched_example() for a
        convenient method for constructing such a dictionary. The dictionary
        contains
        fields.InputDataFields.original_image: [batch_size, H, W, 3] image.
        fields.InputDataFields.original_image_spatial_shape: [batch_size, 2]
          tensor containing the size of the original image.
        fields.InputDataFields.true_image_shape: [batch_size, 3]
          tensor containing the spatial size of the upadded original image.
        fields.InputDataFields.groundtruth_boxes - [batch_size, num_boxes, 4]
          float32 tensor with groundtruth boxes in range [0.0, 1.0].
        fields.InputDataFields.groundtruth_classes - [batch_size, num_boxes]
          int64 tensor with 1-indexed groundtruth classes.
        fields.InputDataFields.groundtruth_instance_masks - (optional)
          [batch_size, num_boxes, H, W] int64 tensor with instance masks.
        fields.DetectionResultFields.detection_boxes - [batch_size,
          max_num_boxes, 4] float32 tensor with detection boxes in range [0.0,
          1.0].
        fields.DetectionResultFields.detection_classes - [batch_size,
          max_num_boxes] int64 tensor with 1-indexed detection classes.
        fields.DetectionResultFields.detection_scores - [batch_size,
          max_num_boxes] float32 tensor with detection scores.
        fields.DetectionResultFields.detection_masks - (optional) [batch_size,
          max_num_boxes, H, W] float32 tensor of binarized masks.
        fields.DetectionResultFields.detection_keypoints - (optional)
          [batch_size, max_num_boxes, num_keypoints, 2] float32 tensor with
          keypoints.

    Returns:
      A dictionary of image summary names to tuple of (value_op, update_op). The
      `update_op` is the same for all items in the dictionary, and is
      responsible for saving a single side-by-side image with detections and
      groundtruth. Each `value_op` holds the tf.summary.image string for a given
      image.
    """
    if self._max_examples_to_draw == 0:
      return {}
    images = self.images_from_evaluation_dict(eval_dict)

    def get_images():
      """Returns a list of images, padded to self._max_images_to_draw."""
      images = self._images
      while len(images) < self._max_examples_to_draw:
        images.append(np.array(0, dtype=np.uint8))
      self.clear()
      return images

    def image_summary_or_default_string(summary_name, image):
      """Returns image summaries for non-padded elements."""
      return tf.cond(
          tf.equal(tf.size(tf.shape(image)), 4),
          lambda: tf.summary.image(summary_name, image),
          lambda: tf.constant(''))

    update_op = tf.py_func(self.add_images, [[images[0]]], [])
    image_tensors = tf.py_func(
        get_images, [], [tf.uint8] * self._max_examples_to_draw)
    eval_metric_ops = {}
    for i, image in enumerate(image_tensors):
      summary_name = self._summary_name_prefix + '/' + str(i)
      value_op = image_summary_or_default_string(summary_name, image)
      eval_metric_ops[summary_name] = (value_op, update_op)
    return eval_metric_ops

  @abc.abstractmethod
  def images_from_evaluation_dict(self, eval_dict):
    """Converts evaluation dictionary into a list of image tensors.

    To be overridden by implementations.

    Args:
      eval_dict: A dictionary with all the necessary information for producing
        visualizations.

    Returns:
      A list of [1, H, W, C] uint8 tensors.
    """
    raise NotImplementedError


class VisualizeSingleFrameDetections(EvalMetricOpsVisualization):
  """Class responsible for single-frame object detection visualizations."""

  def __init__(self,
               category_index,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='Detections_Left_Groundtruth_Right'):
    super(VisualizeSingleFrameDetections, self).__init__(
        category_index=category_index,
        max_examples_to_draw=max_examples_to_draw,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates,
        summary_name_prefix=summary_name_prefix)

  def images_from_evaluation_dict(self, eval_dict):
    return draw_side_by_side_evaluation_image(
        eval_dict, self._category_index, self._max_boxes_to_draw,
        self._min_score_thresh, self._use_normalized_coordinates)
    

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
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
        detection_masks_reframed = reframe_box_masks_to_image_masks(
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

def predict(image_path):
	detection_graph = tf.Graph()
	with detection_graph.as_default():
  		od_graph_def = tf.compat.v1.GraphDef()
  		with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    			serialized_graph = fid.read()
    			od_graph_def.ParseFromString(serialized_graph)
    			tf.import_graph_def(od_graph_def, name='')
	category_index = create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
	
	image = Image.open(image_path)
	# the array based representation of the image will be used later in order to prepare the
	# result image with boxes and labels on it.
	image_np = load_image_into_numpy_array(image)
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image_np, axis=0)
	# Actual detection.
	output_dict = run_inference_for_single_image(image_np, detection_graph)

	# Visualization of the results of a detection.
	visualize_boxes_and_labels_on_image_array(
    		image_np,
    		output_dict['detection_boxes'],
    		output_dict['detection_classes'],
    		output_dict['detection_scores'],
    		category_index,
    		instance_masks=output_dict.get('detection_masks'),
    		use_normalized_coordinates=True,
    		line_thickness=3)
	return image_np

def main(image_path):
  im=Image.fromarray(predict(image_path))
  im.save("newlatest/output_shape.jpg")