#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/logging.h"

using namespace tensorflow;

REGISTER_OP("InnerProductGrad")
	.Input("grad: float32")
	.Input("input: float32")
	.Input("weights: float32")
	.Output("input_grad: float32")
	.Output("weights_grad: float32");

class InnerProductGradOp: public OpKernel {
	public:
		explicit InnerProductGradOp(OpKernelConstruction* context): OpKernel(context){}

	void Compute(OpKernelContext* context) override{

		DCHECK_EQ(3, context->num_inputs());

		const Tensor& grad = context->input(0);
		const Tensor& input = context->input(1);
		const Tensor& weights = context->input(2);

		TensorShape input_shape = input.shape();
		TensorShape weights_shape = weights.shape();

		DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
    	DCHECK_EQ(weights_shape.dim_size(0), grad.shape().dim_size(0));

    	Tensor *input_grad = NULL;
    	Tensor *weights_grad = NULL;

    	OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &input_grad));
    	OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &weights_grad));

		
		auto input_tensor = input.matrix<float>();
		auto weights_tensor = weights.matrix<float>();
		auto grad_tensor = grad.matrix<float>();

		auto grad_inputs_tensor = input_grad->matrix<float>();
		auto grad_weights_tensor = weights_grad->matrix<float>();
		
		for (int i = 0; i < input.shape().dim_size(0); ++i)
   			{
   				grad_inputs_tensor(i,0) = 0;
   				for (int j = 0; j < grad.shape().dim_size(0); ++j)
   				{
   					grad_inputs_tensor(i,0) += grad_tensor(j,0)*weights_tensor(j,i);
   				}
   			}

   		for (int i = 0; i < weights_shape.dim_size(0); ++i)
   			{
   				for (int j = 0; j < weights_shape.dim_size(1); ++j)
   				{
   					grad_weights_tensor(i,j) += grad_tensor(i,0)*input_tensor(j,0);
   				}
   			}
	}
	
};

 
REGISTER_KERNEL_BUILDER(Name("InnerProductGrad").Device(DEVICE_CPU), InnerProductGradOp);
