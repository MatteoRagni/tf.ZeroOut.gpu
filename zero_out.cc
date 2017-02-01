#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

/*   ___       ___          _    _            _   _
/*  / _ \ _ __| _ \___ __ _(_)__| |_ _ _ __ _| |_(_)___ _ _
/* | (_) | '_ \   / -_) _` | (_-<  _| '_/ _` |  _| / _ \ ' \
/*  \___/| .__/_|_\___\__, |_/__/\__|_| \__,_|\__|_\___/_||_|
/*       |_|          |___/
/*/
REGISTER_OP("ZeroOut")
  .Input("to_zero: int32")
  .Output("zeroed: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

/*  _  __                 _ ___       __ _      _ _   _
/* | |/ /___ _ _ _ _  ___| |   \ ___ / _(_)_ _ (_) |_(_)___ _ _
/* | ' </ -_) '_| ' \/ -_) | |) / -_)  _| | ' \| |  _| / _ \ ' \
/* |_|\_\___|_| |_||_\___|_|___/\___|_| |_|_||_|_|\__|_\___/_||_|
/*/
#include "tensorflow/core/framework/op_kernel.h"

void ZeroOutKernelLauncher(const Tensor* t_in, const int n, Tensor* t_out);

class ZeroOutOp : public OpKernel {
public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Tensore in input
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Tensore in output
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->flat<int32>();

#if GOOGLE_CUDA
    ZeroOutKernelLauncher(input, input.size(), output);
#else
    const int N = input.size();
    for (int i = 1; i < N; i++) output(i) = 0;
#endif
    if (N > 0) output(0) = input(0);
  }
};

/*   ___        __     _  __                 _
/*  / _ \ _ __ / _|___| |/ /___ _ _ _ _  ___| |
/* | (_) | '_ \> _|_ _| ' </ -_) '_| ' \/ -_) |
/*  \___/| .__/\_____||_|\_\___|_| |_||_\___|_|
/*       |_|
/*/
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
