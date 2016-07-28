import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Directory for train information output. It will be emptied at the beginning of every run
tf.app.flags.DEFINE_string('log_train_dir', './output/log/train', """Directory where to write train event logs.""")
tf.app.flags.DEFINE_string('log_eval_dir', './output/log/evaluate', """Directory where to write evaluation event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './output/checkpoint', """Directory where to write checkpoint""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Maximum number of training epochs""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement""")
tf.app.flags.DEFINE_integer('batch_size', 300, """Batch size""")
tf.app.flags.DEFINE_integer('image_width', 28, """Width of input images""")
tf.app.flags.DEFINE_integer('image_height', 28, """Height of input images""")
