import os
import os.path as path
import shutil

import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.mesh import summary_v2 as mesh_summary


from datetime import datetime

from dataset.W300 import W300

from model.mobilenetv2 import MobileNetV2


tb_train_dir = "../logs/train"

def main(args):
    # training_set, test_set = compile_dataset_files(args.data)
    w300_dataset = W300(args.data, train=True)

    train_dataset = w300_dataset.create_dataset(args)

    # Tensorboard
    if (os.path.exists(tb_train_dir)):
        shutil.rmtree(tb_train_dir)

    logdir = os.path.join(tb_train_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))

    file_writer = tf.summary.create_file_writer(logdir)

    model = MobileNetV2(input_shape=(256,256,3), k=w300_dataset.expression_number())

    optimizer = tf.keras.optimizers.Adam()
    loss_fn  = tf.keras.losses.mean_squared_error()

    epoch_num = 0
    with file_writer.as_default():
        for step, data in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                img, exp_params, vertex = data

                epoch_step = step % len(w300_dataset)

                if epoch_step == 0:
                    epoch_num += 1
                    print("starting epoch {}".format(epoch_num))


                prediction = model(img, training=True)
                loss = loss_fn(prediction, exp_params)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

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