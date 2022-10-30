import tensorflow as tf
from tensorflow.python.client import device_lib

physical_devices = tf.config.list_physical_devices()
print(tf.test.is_built_with_cuda())
print(physical_devices)

# Test cpu vs gpu
# Make conda for climatea
# conda activate for git bash (doesnt work for powershell)a