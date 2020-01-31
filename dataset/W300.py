import tensorflow as tf
import cv2
import pickle
import glob
import os
import numpy as np

from dataset.W300_parser import *

class W300:
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.isTrain = train
        self.dataset_files = self.compile_dataset_files()

        self.shape_data = parse_shape_data(data_dir)
        self.exp_data = parse_expression_data(data_dir)

        # Average shape and expression
        self.mu = getMeanFaceTensor(self.shape_data, self.exp_data)

        # Blendshapes (29)
        self.exp_shapes = tf.convert_to_tensor(self.exp_data['w_exp'], dtype=tf.float32)

        # Mesh Faces
        self.faces = getMeshFacesTensor(self.shape_data)

    def expression_number(self):
        return 29

    def compile_dataset_files(self):
        data_files = []

        dirs = [d for d in os.listdir(self.data_dir) if d in annotation_dirs]

        for dir in dirs:
            anno_dir = os.path.join(self.data_dir, dir)
            files = [f for f in os.listdir(anno_dir) if f.endswith(annotation_ext)]
            for file in files:
                isTrainSample = 'test' not in file
                if self.isTrain and isTrainSample:
                    data_files.append({
                        'image': os.path.join(anno_dir, img_fn_from_mat_fn(file)),
                        'anno': os.path.join(anno_dir, file),
                    })
                elif not self.isTrain and not isTrainSample:
                    data_files.append({
                        'image': os.path.join(anno_dir, img_fn_from_mat_fn(file)),
                        'anno': os.path.join(anno_dir, file),
                    })

        print("Number of Training Samples: %d" % len(data_files))

        return data_files


    def generator(self):
        for sample in self.dataset_files:
            assert path.isfile(sample['image'])
            img = skio.imread(sample['image'])
            anno = sio.loadmat(sample['anno'])

            img_tf = imgToTensor(img)

            exp_para = tf.convert_to_tensor(anno['Exp_Para'], dtype=tf.float32)
            vertex = applyExpression(self.mu, exp_para, self.exp_shapes)

            yield img_tf, exp_para, vertex

    def getOutputTypes(self):
        return (tf.float32, tf.float32, tf.float32)

    def getOutputShapes(self):
        return (tf.TensorShape([450,450,3]),
                tf.TensorShape([29,1]),
                tf.TensorShape([53215, 3]))

    def __len__(self):
        return len(self.dataset_files)

    def create_dataset(self, args):
        dataset = tf.data.Dataset.from_generator(generator=self.generator,
                                                 output_types=self.getOutputTypes(),
                                                 output_shapes=self.getOutputShapes())

        # This dataset will run for a specified number of epochs
        dataset = dataset.repeat(args.epochs)

        # Set the number of datapoints you want to load and shuffle
        dataset = dataset.shuffle(args.no_shuffle)

        # Set the batchsize
        dataset = dataset.batch(args.train_batch)

        return dataset

def main(args):
    # training_set, test_set = compile_dataset_files(args.data)
    w300_dataset = W300(args.data, train=True)

    train_dataset = w300_dataset.create_dataset(args)

    # Tensorboard
    if (os.path.exists(W300_log_dir)):
        shutil.rmtree(W300_log_dir)

    logdir = os.path.join(W300_log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))

    file_writer = tf.summary.create_file_writer(logdir)

    epoch_num = 0
    with file_writer.as_default():
        for step, data in enumerate(train_dataset):
            img, exp_params, vertex = data

            epoch_step = step % len(w300_dataset)

            if epoch_step == 0:
                epoch_num += 1
                print("starting epoch {}".format(epoch_num))

            if step % args.vis_freq == 0:
                print("epoch {} step {}".format(epoch_num, epoch_step))

                tf.summary.image("Training Image", img, max_outputs=1, step=step)
                summary = mesh_summary.mesh('GT Expression Mesh',
                                            vertices=vertex[0:],
                                            faces=w300_dataset.faces,
                                            step=step)
                file_writer.flush()



if __name__ == '__main__':
    from utils import opts
    args = opts.argparser()

    main(args)
