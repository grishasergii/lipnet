import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('logdir', './output/log', """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './output/checkpoint', """Directory where to write checkpoint""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Maximum number of training epochs""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement""")
tf.app.flags.DEFINE_integer('image_width', 28, """Width of input images""")
tf.app.flags.DEFINE_integer('image_height', 28, """Height of input images""")
