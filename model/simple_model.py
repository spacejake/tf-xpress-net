import tensorflow as tf


class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.Conv2D(32, 3, activation='relu')
    self.flatten = tf.keras.Flatten()
    self.d1 = tf.keras.Dense(128, activation='relu')
    self.d2 = tf.keras.Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)