#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

Tensor empty_mkldnn(IntArrayRef sizes, const TensorOptions& options, c10::optional<c10::MemoryFormat> optional_memory_format) {
  //TORCH_CHECK(
  //   !options.has_memory_format(),
  //   "'memory_format' argument is incompatible with mkldnn tensor");
  TORCH_CHECK(
     !optional_memory_format.has_value(),
     "'memory_format' argument is incompatible with mkldnn tensor");
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  ideep::tensor::dims dst_dims (sizes.begin(), sizes.end());
  //ideep::tensor it {dst_dims, ideep::tensor::data_type::f32};
  ideep::tensor::data_type ideep_tensor_data_type;
  if(options.dtype() == at::kFloat){
    ideep_tensor_data_type = ideep::tensor::data_type::f32;
  }else if(options.dtype() == at::kBFloat16){
    ideep_tensor_data_type = ideep::tensor::data_type::bf16;
  }else if(options.dtype() == ScalarType::Byte){
    ideep_tensor_data_type = ideep::tensor::data_type::u8;
  }else{
  	TORCH_WARN("LeslieDebug options.dtype() is: ", options.dtype());
    TORCH_CHECK(false, "empty_mkldnn expects float or bfloat16 tensor input");
  }
  ideep::tensor it {dst_dims, ideep_tensor_data_type};
  return new_with_itensor_mkldnn(std::move(it), options);
}

#else

Tensor empty_mkldnn(IntArrayRef sizes, const TensorOptions& options, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(false, "empty_mkldnn: MKL-DNN build is disabled");
}

#endif // AT_MKLDNN_ENABLED()

}}
