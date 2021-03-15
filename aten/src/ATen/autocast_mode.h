#pragma once

namespace at {
namespace autocast {

#define ADD_NS(RAW_OP) at::RAW_OP

TORCH_API bool is_enabled();
TORCH_API void set_enabled(bool enabled);
TORCH_API at::ScalarType get_dtype();
TORCH_API void set_dtype(at::ScalarType);
TORCH_API void clear_cache();
TORCH_API int increment_nesting();
TORCH_API int decrement_nesting();

TORCH_API extern int cpu_dtype;
TORCH_API extern int cpu_device;


using weakref_type = c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
using val_type = std::tuple<weakref_type, Tensor>;

TORCH_API extern thread_local std::unordered_map<TensorImpl*, val_type> cached_casts_cpu;

//TORCH_API int get_input_dtype_priority();
//TORCH_API int get_input_device_priority();

// Policies correspond to op categories that need code-divergent handling.
// Wrapper templates below are specialized based on a policy template parameter.
enum class CastPolicy : uint8_t {
  fp16 = 0, // Cast all inputs to at::kHalf before running the op.
  fp32, // Cast all inputs to at::kFloat before running the op.
  fp32_set_opt_dtype, // Treats functions (like softmax) that
                      //   1. we'd like to run in fp32 and
                      //   2. have a c10::optional<ScalarType> arg that controls the output type.
                      // fp32_set_opt_dtype wrappers' policy is:  if the output type is already set,
                      // don't touch it, otherwise, set it to at::kFloat.
  fp32_append_dtype, // Treats functions (like norm) that
                     //   1. we'd like to run in fp32 and
                     //   2. have some overloads that accept an output type and other overloads that don't.
                     // fp32_append_dtype wrappers wrap the overloads that don't have an output dtype.
                     // The wrapper policy is:  append at::kFloat to the args, and redispatch to the
                     // type-aware overload.
  promote, // Run in the widest dtype among several args.
  runtime,
};

/********************************************************************
Logic to extract the promote type from any Tensor or TensorList args.
********************************************************************/

// Overload to catch Tensor args.
// If nextArg is floating-point, compare its scalar_type with our
// current best guess for the promote type, and update if necessary.
inline at::ScalarType prioritize(at::ScalarType current, const Tensor& nextArg) {
  if (current == at::kDouble) {
    AT_ERROR("promote type is double in at::autocast::prioritize");
    return current;
  }
  if (nextArg.is_cuda() && nextArg.is_floating_point()) {
    auto next = nextArg.scalar_type();
    if (next == at::kDouble) {
      return current; // ignores double tensors
    } else if (current == at::kFloat || next == at::kFloat) {
      return at::kFloat; // prioritizes float over half
    } else if (current == at::kHalf && next == at::kHalf) {
      return at::kHalf;
    } else {
      AT_ERROR("Unexpected floating ScalarType in at::autocast::prioritize");
      return current;
    }
  } else {
    return current;
  }
}

// Overload to catch TensorList args (for e.g. cat, stack).
// Reuses the overload above to process each Tensor in the list.
inline at::ScalarType prioritize(at::ScalarType current, const TensorList& list) {
  for (const auto& tensor : list) {
    current = prioritize(current, tensor);
  }
  return current;
}

// Template to catch non-Tensor args (no-op that returns current best guess)
template<typename T>
inline at::ScalarType prioritize(at::ScalarType current, T nextArg) {
  return current;
}

// Overload for the tail case.
inline at::ScalarType promote_type(at::ScalarType current) {
  return current;
}

// Unpack args and determine if incoming float16 tensors need to be promoted to float32.
// Non-Tensor arguments are ignored.
template<typename Arg0, typename... Args>
inline at::ScalarType promote_type(at::ScalarType current, Arg0 arg0, Args... args) {
  auto new_current = prioritize(current, arg0);
  return promote_type(new_current, args...);
}

/****************************************************
Logic to apply cached casting to any Tensor argument.
****************************************************/
inline bool is_eligible(const Tensor& arg) {
  return (arg.defined() && arg.is_floating_point() && (arg.scalar_type() != at::kDouble));
}

// Overload to catch Tensor args
TORCH_API Tensor cached_cast(at::ScalarType to_type, const Tensor& arg);

// Overload to process optional<Tensor>
inline c10::optional<Tensor> cached_cast(at::ScalarType to_type, const c10::optional<Tensor>& arg) {
  if (arg.has_value()) {
    return cached_cast(to_type, *arg);
  } else {
    return c10::nullopt;
  }
}

// Overload to process TensorLists
inline std::vector<Tensor> cached_cast(at::ScalarType to_type, const TensorList& arg) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.push_back(cached_cast(to_type, t));
  }
  return vec;
}

// Template to catch non-Tensor args.
template<typename T>
inline T cached_cast(at::ScalarType to_type, T arg) {
  return arg;
}

/*******************************************************
Logic to flip an output dtype flag.
Keep it simple for now by assuming only one such flag is
present in the argument list.  If I ever need a function
with more than flag I'll figure out something else.
The policy is:
If the user has explicity specified a dtype, respect it.
Otherwise, set it to the autocast type.
********************************************************/

// Overload to catch dtype flags
c10::optional<ScalarType> inline set_opt_dtype(at::ScalarType to_type, const c10::optional<ScalarType>& dtype) {
  return dtype.has_value() ? dtype : to_type;
}

// Template to catch other args
template<typename T>
inline T set_opt_dtype(at::ScalarType to_type, T arg) {
  return arg;
}

template<typename... Args>
inline bool firstarg_is_eligible(const Tensor& arg, Args... args) {
  return is_eligible(arg);
}

template<typename... Args>
inline at::ScalarType type_from_firstarg(at::ScalarType to_type, const Tensor& arg, Args... args) {
  return (is_eligible(arg) ? to_type : arg.scalar_type());
}

} // namespace autocast
} // namespace at
