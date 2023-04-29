using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Examples.Utils
{
    public static class VBHelper
    {
        public static float AsSingle(this torch.Tensor tensor)
        {
            return tensor.item<float>();
        }
        
        public static long AsLong(this torch.Tensor tensor)
        {
            return tensor.item<long>();
        }

        /// <summary>
        /// Expects input to be 1- or 2-D tensor and transposes dimensions 0 and 1.
        /// </summary>
        public static torch.Tensor Transpose2D(this torch.Tensor tensor)
        {
            return tensor.t();
        }

        public static Tensor GetTensor(long[] dataArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return torch.tensor(dataArray, dtype, device, requires_grad, names);
        }

        public static torch.Device CudaDevice => torch.CUDA;
    }
}
