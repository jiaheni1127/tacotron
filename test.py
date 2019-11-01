import tensorflow as tf
from models import create_model
from util import audio
from hparams import hparams

checkpoint_path = "/tacotron-20180906/model.ckpt"
model_name='tacotron'
inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
with tf.variable_scope('model') as scope:
    model = create_model(model_name, hparams)
    model.initialize(inputs, input_lengths)
    wav_output = audio.inv_spectrogram_tensorflow(model.linear_outputs[0])

print('Loading checkpoint: %s' % checkpoint_path)

session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint_path)

with open("eval.pb", 'w') as f:
    g = tf.get_default_graph()
    f.write(str(g.as_graph_def()))
