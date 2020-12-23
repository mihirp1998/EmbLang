# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal data reader for GQN TFRecord datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import constants as const
import numpy as np
import utils
import collections
import os
import tensorflow as tf
nest = tf.contrib.framework.nest
from ipdb import set_trace as st

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])

_counter = 0

_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    #originally 2160 and 240 -> 500 and 40
    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=500,
        test_size=40,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)

_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5
_MODES = ('train', 'test')


def _get_dataset_files(dateset_info, mode, root):
  """Generates lists of files for a given dataset version."""
  basepath = dateset_info.basepath
  base = os.path.join(root, basepath, mode)
  if mode == 'train':
    num_files = dateset_info.train_size
  else:
    num_files = dateset_info.test_size

  length = len(str(num_files))

  #dirty hack:  
  if const.GQN_DATA_NAME == 'rooms_ring_camera':
    length += 1
    if mode == 'test':
      num_files_ = 240
    else:
      num_files_ = 2160
  else:
    num_files_ = num_files
    
  template = '{:0%d}-of-{:0%d}.tfrecord' % (length, length)

  #1 indexed!!!
  return [os.path.join(base, template.format(i, num_files_))
          for i in range(1, num_files+1)]


def _convert_frame_data(jpeg_data):
  decoded_frames = tf.image.decode_jpeg(jpeg_data)
  return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


class DataReader(object):
  """Minimal queue based TFRecord reader.

  You can use this reader to load the datasets used to train Generative Query
  Networks (GQNs) in the 'Neural Scene Representation and Rendering' paper.
  See README.md for a description of the datasets and an example of how to use
  the reader.
  """

  def __init__(self,
               dataset,
               context_size,
               root,
               mode='train',
               # Optionally reshape frames
               custom_frame_size=None,
               # Queue params
               num_threads=4,
               capacity=256,
               min_after_dequeue=128,
               seed=0):
    """Instantiates a DataReader object and sets up queues for data reading.

    Args:
      dataset: string, one of ['jaco', 'mazes', 'rooms_ring_camera',
          'rooms_free_camera_no_object_rotations',
          'rooms_free_camera_with_object_rotations', 'shepard_metzler_5_parts',
          'shepard_metzler_7_parts'].
      context_size: integer, number of views to be used to assemble the context.
      root: string, path to the root folder of the data.
      mode: (optional) string, one of ['train', 'test'].
      custom_frame_size: (optional) integer, required size of the returned
          frames, defaults to None.
      num_threads: (optional) integer, number of threads used to feed the reader
          queues, defaults to 4.
      capacity: (optional) integer, capacity of the underlying
          RandomShuffleQueue, defualts to 256.
      min_after_dequeue: (optional) integer, min_after_dequeue of the underlying
          RandomShuffleQueue, defualts to 128.
      seed: (optional) integer, seed for the random number generators used in
          the reader.

    Raises:
      ValueError: if the required version does not exist; if the required mode
         is not supported; if the requested context_size is bigger than the
         maximum supported for the given dataset version.
    """

    if dataset not in _DATASETS:
      raise ValueError('Unrecognized dataset {} requested. Available datasets '
                       'are {}'.format(dataset, _DATASETS.keys()))

    if mode not in _MODES:
      raise ValueError('Unsupported mode {} requested. Supported modes '
                       'are {}'.format(mode, _MODES))

    self._dataset_info = _DATASETS[dataset]

    if context_size >= self._dataset_info.sequence_size:
      raise ValueError(
          'Maximum support context size for dataset {} is {}, but '
          'was {}.'.format(
              dataset, self._dataset_info.sequence_size-1, context_size))

    self._context_size = context_size
    # Number of views in the context + target view
    self._example_size = context_size + 1
    self._custom_frame_size = custom_frame_size

    with tf.device('/cpu'):
      file_names = _get_dataset_files(self._dataset_info, mode, root)
      filename_queue = tf.compat.v1.train.string_input_producer(file_names, seed=seed)
      reader = tf.compat.v1.TFRecordReader()

      if mode == 'test':
        num_threads = 1
        
      read_ops = [self._make_read_op(reader, filename_queue)
                  for _ in range(num_threads)]

      dtypes = nest.map_structure(lambda x: x.dtype, read_ops[0])
      shapes = nest.map_structure(lambda x: x.shape[1:], read_ops[0])

      if mode == 'train': 
        self._queue = tf.queue.RandomShuffleQueue(
          capacity=capacity,
          min_after_dequeue=min_after_dequeue,
          dtypes=dtypes,
          shapes=shapes,
          seed=seed)
      else:
        self._queue = tf.queue.FIFOQueue(
          capacity=capacity,
          dtypes=dtypes,
          shapes=shapes)

      enqueue_ops = [self._queue.enqueue_many(op) for op in read_ops]
      tf.compat.v1.train.add_queue_runner(tf.compat.v1.train.QueueRunner(self._queue, enqueue_ops))

  def read(self, batch_size):
    """Reads batch_size (query, target) pairs."""

    if const.generate_views:
      return self.read_gen(batch_size)
    
    frames, cameras = self._queue.dequeue_many(batch_size)
    return self.read_with_frames_and_cams(frames, cameras)

  def read_with_frames_and_cams(self, frames, cameras):
    context_frames = frames[:, :-1]
    context_cameras = cameras[:, :-1]
    target = frames[:, -1]
    query_camera = cameras[:, -1]
    context = Context(cameras=context_cameras, frames=context_frames)
    query = Query(context=context, query_camera=query_camera)
    return TaskData(query=query, target=target)
  
  def read_gen(self, bs):
    assert bs == 1
    frames, cameras = self._queue.dequeue_many(bs)
    #frames_ref = tf.Variable(0, shape = frames.shape)
    #cameras_ref = tf.Variable(0, shape = cameras.shape)

    init_ref = lambda x: tf.Variable(lambda: tf.zeros_like(x), dtype = tf.float32)
    frames_ref = init_ref(frames)
    cameras_ref = init_ref(cameras)
    
    counter = tf.Variable(0, dtype = tf.int32)
    increment_op = tf.compat.v1.assign_add(counter, 1)
    counter_mod = tf.mod(counter, const.GEN_NUM_VIEWS)

    assign_if_zero = lambda x, y: tf.cond(pred=tf.equal(counter_mod, 0), true_fn=lambda: tf.compat.v1.assign(x, y), false_fn=lambda: x)
    frame_op = assign_if_zero(frames_ref, frames)
    camera_op = assign_if_zero(cameras_ref, cameras)
    
    update_ops = [camera_op, frame_op, increment_op]
    with tf.control_dependencies(update_ops):
      return self.read_with_frames_and_cams(frames_ref, cameras_ref)

  def _make_read_op(self, reader, filename_queue):
    """Instantiates the ops used to read and parse the data into tensors."""
    _, raw_data = reader.read_up_to(filename_queue, num_records=16)
    feature_map = {
        'frames': tf.io.FixedLenFeature(
            shape=self._dataset_info.sequence_size, dtype=tf.string),
        'cameras': tf.io.FixedLenFeature(
            shape=[self._dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS],
            dtype=tf.float32)
    }
    example = tf.io.parse_example(serialized=raw_data, features=feature_map)
    indices = self._get_randomized_indices()
    frames = self._preprocess_frames(example, indices)
    cameras = self._preprocess_cameras(example, indices)
    return frames, cameras

  def _get_randomized_indices(self):
    global _counter
    _counter += 1
    #print('seed is', _counter)
    """Generates randomized indices into a sequence of a specific length."""
    indices = tf.range(0, self._dataset_info.sequence_size)
    indices = tf.random.shuffle(indices, seed = _counter)
    indices = tf.slice(indices, begin=[0], size=[self._example_size])
    return indices

  def _preprocess_frames(self, example, indices):
    """Instantiates the ops used to preprocess the frames data."""
    frames = tf.concat(example['frames'], axis=0)
    frames = tf.gather(frames, indices, axis=1)
    frames = tf.map_fn(
        _convert_frame_data, tf.reshape(frames, [-1]),
        dtype=tf.float32, back_prop=False)
    dataset_image_dimensions = tuple(
        [self._dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
    frames = tf.reshape(
        frames, (-1, self._example_size) + dataset_image_dimensions)
    if (self._custom_frame_size and
        self._custom_frame_size != self._dataset_info.frame_size):
      frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
      new_frame_dimensions = (self._custom_frame_size,) * 2 + (_NUM_CHANNELS,)
      frames = tf.image.resize(
          frames, new_frame_dimensions[:2], method=tf.image.ResizeMethod.BILINEAR)
      frames = tf.reshape(
          frames, (-1, self._example_size) + new_frame_dimensions)
    return frames

  def _preprocess_cameras(self, example, indices):
    """Instantiates the ops used to preprocess the cameras data."""
    raw_pose_params = example['cameras']
    raw_pose_params = tf.reshape(
        raw_pose_params,
        [-1, self._dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
    raw_pose_params = tf.gather(raw_pose_params, indices, axis=1)
    pos = raw_pose_params[:, :, 0:3]
    yaw = raw_pose_params[:, :, 3:4]
    pitch = raw_pose_params[:, :, 4:5]
    
    # cameras = tf.concat(
    #    [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)

    # i just want yaw and pitch
    #return tf.concat([pos, yaw, pitch], axis = 2)

    return tf.concat([-pitch, np.pi-yaw], axis = 2)
  
    # (bad)
    #return tf.concat([np.pi-yaw, -pitch], axis = 2)
    

    #let's try flipping pitch? (also bad)
    #return tf.concat([np.pi-yaw, pitch], axis = 2)

    #or flipping yaw? (bad)
    #return tf.concat([yaw, -pitch], axis = 2)

    #or both? (bad)
    #return tf.concat([yaw, pitch], axis = 2) 
