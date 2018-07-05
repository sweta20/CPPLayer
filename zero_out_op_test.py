import tensorflow as tf


class ZeroOutTest(tf.test.TestCase):
	def testZeroOut(self):
		zero_out_module = tf.load_op_library("./zero_out.so")
		with self.test_session():
			result = zero_out_module.zero_out([[1,2],[3,4]])
			self.assertAllEqual(result.eval(), [[1,0],[0,0]])

if __name__ == "__main__":
	tf.test.main()