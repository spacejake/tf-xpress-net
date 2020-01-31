import os
import os.path as path
import shutil

import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.mesh import summary_v2 as mesh_summary

import numpy as np
import sklearn.metrics
import scipy.io as sio
import skimage.io as skio
import cv2

import matplotlib.pyplot as plt
from datetime import datetime
import hashlib
import time

from dataset.util.tf_record_utils import *

W300_log_dir = "../logs/test/dataset/300W"

tfrecod_filenames = {
  "train": "300W-LP-train.tfrecord",
  "test": "300W-LP-test.tfrecord",
}

annotation_dirs = ['AFW', 'HELEN', 'IBUG', 'LFPW']
annotation_ext = '_0.mat'
image_ext = '_0.jpg'

def load_image(img_path):
    assert path.isfile(img_path)

    img = skio.imread(img_path)
    return img

def imgToTensor(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    if tf.reduce_max(img) > 1:
        img /= 255
    return img

def imshow(img):
    plt.imshow(img)
    plt.axis('off')

def parse_shape_data(data_dir):
    fn = path.join(data_dir, 'Code', 'Model_Shape.mat')
    return sio.loadmat(fn)

def parse_expression_data(data_dir):
    fn = path.join(data_dir, 'Code', 'Model_Exp.mat')
    return sio.loadmat(fn)

def applyExpression(mu, exp_params, exp_shapes):
    vertex = mu + tf.matmul(exp_shapes, exp_params)
    vertex = tf.reshape(vertex, (vertex.shape[0] // 3, 3))
    # vertex = vertex * tf.constant([[[1.0, 1.0, -1.0]]])
    return vertex

def getMeanFaceTensor(shape_data, exp_data):
    mu = tf.convert_to_tensor(shape_data['mu_shape'], dtype=tf.float32) \
         + tf.convert_to_tensor(exp_data['mu_exp'], dtype=tf.float32)
    return mu

def getMeshFacesTensor(shape_data):
    faces = tf.expand_dims(tf.transpose(tf.convert_to_tensor(shape_data['tri'], dtype=tf.int32)), 0)

    # Reverse order of face
    faces = faces[:, :, ::-1]

    # Faces are 1-indexed (Matlab) not 0-indexed like python, convert to 0 index
    faces = faces - 1
    return faces


def parse_sample_anno(data_dir, mat_fn):
    mat_path = path.join(data_dir, mat_fn)
    anno_mat = sio.loadmat(mat_path)

    return anno_mat

def parse_sample_image(data_dir, mat_fn):
    image_fn = img_fn_from_mat_fn(mat_fn)
    image_path = path.join(data_dir, image_fn)

    img = load_image(image_path)
    return img

def img_fn_from_mat_fn(mat_fn):
    return mat_fn[:-len(annotation_ext)] + image_ext

def to_tfrecord(img_path, expression_params, reconstructed_verts):
    img = cv2.imread(img_path)

    # encoded_image_data_old = open(filepath, 'rb').read()
    encoded_image_data = cv2.imencode('.jpg', img)[1].tostring()
    key = hashlib.sha256(encoded_image_data).hexdigest()

    height, width, channel = img.shape
    print("height is %d, width is %d, channel is %d" % (height, width, channel))

    tfrecord = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(int(height)),
        'image/width': int64_feature(int(width)),
        'image/filename': bytes_feature(img_path.encode('utf8')),
        'image/source_id': bytes_feature(img_path.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_image_data),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/exp_params/size': int64_list_feature(expression_params.shape),
        'image/object/exp_params/data': float_list_feature(tf.reshape(expression_params, [-1])),
        'image/object/vertices/size': int64_list_feature(reconstructed_verts.shape),
        'image/object/vertices/data': float_list_feature(tf.reshape(reconstructed_verts, [-1])),
    }))

    return tfrecord

def serialize(img_path, expression_params, reconstructed_verts):
    img = cv2.imread(img_path)

    # encoded_image_data_old = open(filepath, 'rb').read()
    encoded_image_data = cv2.imencode('.jpg', img)[1].tostring()
    key = hashlib.sha256(encoded_image_data).hexdigest()

    height, width, channel = img.shape
    feature_dict = {
        'image/height': height,
        'image/width': width,
        'image/channels': channel,
        'image/filename': img_path.encode('utf8'),
        'image/source_id': img_path.encode('utf8'),
        'image/key/sha256': key.encode('utf8'),
        'image/encoded': encoded_image_data,
        'image/format': 'jpeg'.encode('utf8'),
        'image/object/exp_params/data': tf.io.serialize_tensor(expression_params).numpy(),
        'image/object/vertices/data': tf.io.serialize_tensor(reconstructed_verts).numpy(),
    }


def convert_pickle(args):
    shape_data = parse_shape_data(args.data)
    exp_data = parse_expression_data(args.data)

    # Average shape and expression
    mu = getMeanFaceTensor(shape_data, exp_data)

    # Blendshapes (29)
    w_exp = tf.convert_to_tensor(exp_data['w_exp'], dtype=tf.float32)

    # Mesh Faces
    # faces = getMeshFacesTensor(shape_data)

    dirs = [d for d in os.listdir(args.data) if d in annotation_dirs]

    train_samples = 0
    test_samples = 0

    for dir in dirs:
        anno_dir = os.path.join(args.data, dir)
        files = [f for f in os.listdir(anno_dir) if f.endswith(annotation_ext)]
        for file in files:
            isTrain = 'test' not in file
            img_path = path.join(anno_dir, img_fn_from_mat_fn(file))

            anno = parse_sample_anno(anno_dir, file)

            exp_para = tf.convert_to_tensor(anno['Exp_Para'], dtype=tf.float32)
            vertex = applyExpression(mu, exp_para, w_exp)

def convert_TFRecords(args):
    writer_train = tf.io.TFRecordWriter(os.path.join(args.data, tfrecod_filenames['train']))
    writer_test = tf.io.TFRecordWriter(os.path.join(args.data, tfrecod_filenames['test']))

    shape_data = parse_shape_data(args.data)
    exp_data = parse_expression_data(args.data)

    # Average shape and expression
    mu = getMeanFaceTensor(shape_data, exp_data)

    # Blendshapes (29)
    w_exp = tf.convert_to_tensor(exp_data['w_exp'], dtype=tf.float32)

    # Mesh Faces
    # faces = getMeshFacesTensor(shape_data)

    dirs = [d for d in os.listdir(args.data) if d in annotation_dirs]

    train_samples = 0
    test_samples = 0

    for dir in dirs:
        anno_dir = os.path.join(args.data, dir)
        files = [f for f in os.listdir(anno_dir) if f.endswith(annotation_ext)]
        for file in files:
            isTrain = 'test' not in file
            img_path = path.join(anno_dir, img_fn_from_mat_fn(file))

            anno = parse_sample_anno(anno_dir, file)

            exp_para = tf.convert_to_tensor(anno['Exp_Para'], dtype=tf.float32)
            vertex = applyExpression(mu, exp_para, w_exp)

            tf_sample = to_tfrecord(img_path, exp_para, vertex)

            if isTrain:
                train_samples += 1
                print("Writing Training Sample: %d" % train_samples)
                writer_train.write(tf_sample.SerializeToString())
            else:
                test_samples += 1
                print("Writing Test Sample: %d" % test_samples)
                writer_test.write(tf_sample.SerializeToString())

            if (train_samples > 5): break
        if (train_samples > 5): break

    writer_train.close()
    writer_test.close()

    print("Complete!!")
    print("Number of Training Samples: %d" % train_samples)
    print("Number of Test Samples: %d" % test_samples)


def main(args):
    # Clear out any prior log data.

    # Sets up a timestamped log directory.

    if (os.path.exists(W300_log_dir)):
        shutil.rmtree(W300_log_dir)

    logdir = os.path.join(W300_log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)

    shape_data = parse_shape_data(args.data)
    exp_data = parse_expression_data(args.data)

    # Average shape and expression
    mu = getMeanFaceTensor(shape_data, exp_data)

    # Blendshapes (29)
    w_exp = tf.convert_to_tensor(exp_data['w_exp'], dtype=tf.float32)

    # Mesh Faces
    faces = getMeshFacesTensor(shape_data)

    with file_writer.as_default():
        dirs = [d for d in os.listdir(args.data) if d in annotation_dirs]
        for dir in dirs:
            anno_dir = os.path.join(args.data, dir)
            files = [f for f in os.listdir(anno_dir) if f.endswith(annotation_ext)]
            for file in files:
                isTest = 'test' in file
                anno = parse_sample_anno(anno_dir, file)
                img = parse_sample_image(anno_dir, file)

                imshow(img)
                img_tf = imgToTensor(img)

                exp_Para = tf.convert_to_tensor(anno['Exp_Para'], dtype=tf.float32)

                vertex = applyExpression(mu, exp_Para, w_exp)

                tf.summary.image("Training Image", tf.expand_dims(img_tf, 0), max_outputs=25, step=0)
                summary = mesh_summary.mesh('Expression Mesh', vertices=tf.expand_dims(vertex, 0), faces=faces, step=0)
                file_writer.flush()


if __name__ == '__main__':
    from utils import opts
    args = opts.argparser()

    main(args)
