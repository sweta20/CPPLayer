import unittest
import numpy as numpy
import tensorflow as tf 
import _inner_product_grad
import numpy as np
inner_product_module = tf.load_op_library("inner_product.so")

class InnerProductOpTest(unittest.TestCase):
	def test_raisesExceptionWithIncompatibleDimensions(self):
		with tf.Session(""):
			with self.assertRaises(ValueError):
				inner_product_module.inner_product([1,2], [[1,2], [3,4]]).eval()
			with self.assertRaises(ValueError):
				self.assertRaises(inner_product_module.inner_product([1, 2], [1, 2, 3, 4]).eval(), ValueError)
			with self.assertRaises(ValueError):
				self.assertRaises(inner_product_module.inner_product([1, 2, 3], [[1, 2], [3, 4]]).eval(), ValueError)

	def test_innerProductHardCoded(self):
		with tf.Session(""):
			result = inner_product_module.inner_product([[1], [2]], [[1,2], [3,4]]).eval()
			self.assertEqual(result.shape[0], 2)
			self.assertEqual(result[0], 5)
			self.assertEqual(result[1], 11)

	def test_innerProductGradientXHardCoded(self):
		with tf.Session("") as sess:

			x = tf.placeholder(tf.float32, shape=(2))
			W = tf.constant(np.asarray([[1,2],[3,4]]).astype(np.float32))

			Wx_tf = tf.matmul(W, tf.reshape(x, [-1,1]))
			Wx_inner_product = inner_product_module.inner_product(tf.reshape(x, [-1,1]), W)

			grad_x_tf = tf.gradients(Wx_tf, x)
			grad_x_inner_product = tf.gradients(Wx_inner_product, x)

			gradient_tf = sess.run(grad_x_tf, feed_dict = {x:np.asarray([1,2]).astype(np.float32)})
			gradient_inner_product= sess.run(grad_x_inner_product, feed_dict = {x:np.asarray([1,2]).astype(np.float32)})
			
			self.assertEqual(gradient_tf[0][0], gradient_inner_product[0][0])
			self.assertEqual(gradient_tf[0][1], gradient_inner_product[0][1])

	def test_innerProductGradientWHardCoded(self):
		with tf.Session("") as sess:

			x = tf.constant(np.asarray([1,2]).astype(np.float32))
			W = tf.placeholder(tf.float32, shape= (2,2))

			Wx_tf = tf.matmul(W, tf.reshape(x, [-1,1]))
			Wx_inner_product = inner_product_module.inner_product(tf.reshape(x, [-1,1]), W)

			grad_W_tf = tf.gradients(Wx_tf, W)
			grad_W_inner_product = tf.gradients(Wx_inner_product, W)

			gradient_tf = sess.run(grad_W_tf, feed_dict = {W:np.asarray([[1,2],[3,4]]).astype(np.float32)})
			gradient_inner_product= sess.run(grad_W_inner_product, feed_dict = {W:np.asarray([[1,2],[3,4]]).astype(np.float32)})
			
			self.assertEqual(gradient_tf[0][0][0], gradient_inner_product[0][0][0])
			self.assertEqual(gradient_tf[0][0][1], gradient_inner_product[0][0][1])
			self.assertEqual(gradient_tf[0][1][0], gradient_inner_product[0][1][0])
			self.assertEqual(gradient_tf[0][1][1], gradient_inner_product[0][1][1])


if __name__ == '__main__':
	unittest.main()
