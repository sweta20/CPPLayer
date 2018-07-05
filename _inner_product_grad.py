from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import tensorflow as tf 

inner_product_grad_module = tf.load_op_library("inner_product_grad.so")

@ops.RegisterGradient("InnerProduct")
def _inner_product_grad_cc(op, grad):
	return inner_product_grad_module.inner_product_grad(grad, op.inputs[0], op.inputs[1])

def _inner_product_grad(op, grad):
	input_tensor = op.inputs[0]
	weights_tensor = op.inputs[1]

	input_rows = array_ops.shape(input_tensor)[0]
	output_rows = array_ops.shape(weights_tensor)[0]

	grad_input = tf.matmul(tf.transpose(grad), weights_tensor)
	grad_weights = tf.matmul(tf.transpose(grad),  tf.reshape(tf.tile(tf.reshape(input_tensor, [input_rows]), [output_rows]), [output_rows, -1]))
	
	return [tf.transpose(grad_input), grad_weights]