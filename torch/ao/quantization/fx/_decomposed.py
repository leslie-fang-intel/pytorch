import torch
from torch.library import Library, impl
from torch.ao.quantization import MinMaxObserver
from typing import Tuple

# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_decomposed", "DEF")

_DTYPE_TO_QVALUE_BOUNDS = {
    torch.uint8: (0, 255),
    torch.int8: (-128, 127),
    torch.int32: (-(2**31), 2**31 - 1)
}

# Helper to check the passed in quant min and max are valid for the dtype
def _quant_min_max_bounds_check(quant_min, quant_max, dtype):
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]

    assert quant_min >= quant_min_lower_bound, \
        "quant_min out of bound for dtype, " \
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"

    assert quant_max <= quant_max_upper_bound, \
        "quant_max out of bound for dtype, " \
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"

quantized_decomposed_lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_per_tensor(
        input: torch.Tensor,
        scale: float,
        zero_point: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """ Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 Tensor
       scale (float): quantization parameter for affine quantization
       zero_point (int): quantization parameter for affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)

    inv_scale = 1.0 / scale
    return torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype)

quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "CompositeExplicitAutograd")
def quantize_per_tensor_tensor(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """ Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    assert zero_point.numel() == 1, f"Exepecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Exepecting scale tensor to be one element, but received : {scale.numel()}"
    return quantize_per_tensor(input, scale.item(), zero_point.item(), quant_min, quant_max, dtype)

@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "Meta")
def quantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    assert zero_point.numel() == 1, f"Exepecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Exepecting scale tensor to be one element, but received : {scale.numel()}"
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)

# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor(
        input: torch.Tensor,
        scale: float,
        zero_point: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """ Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per tensor quantized Tensor if combined with
       quantization parameters in the argument of this function (scale/zero_point)

       scale (float): quantization parameter for affine quantization

       zero_point (int): quantization parameter for affine quantization

       quant_min (int): minimum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): dtype for input Tensor (not used in computation,
       reserved for pattern matching)

    Returns:
       dequantized float32 Tensor
    """
    assert input.dtype == dtype, f"Expecting input to have dtype: {dtype}"
    if dtype in [torch.uint8, torch.int8, torch.int32]:
        # TODO: investigate why
        # (input - zero_point).to(torch.float32) * scale
        # failed the test
        return (input.to(torch.float32) - zero_point) * scale
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")


quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor_tensor(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """ Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    assert zero_point.numel() == 1, f"Exepecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Exepecting scale tensor to be one element, but received : {scale.numel()}"
    return dequantize_per_tensor(input, scale.item(), zero_point.item(), quant_min, quant_max, dtype)

@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "Meta")
def dequantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    assert zero_point.numel() == 1, f"Exepecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    assert scale.numel() == 1, f"Exepecting scale tensor to be one element, but received : {scale.numel()}"
    assert input.dtype == dtype, f"Expecting input to have dtype: {dtype}"
    if dtype in [torch.uint8, torch.int8, torch.int32]:
        return torch.empty_like(input, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")

from typing import List, Optional, Union
from torch._prims_common import (
    check,
    corresponding_complex_dtype,
    corresponding_real_dtype,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    FloatLike,
    IntLike,
    make_contiguous_strides_for,
)

def calc_conv_nd_return_shape(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: Union[List[int], int],
    padding: Union[List[int], int],
    dilation: Union[List[int], int],
    is_transposed: bool,
    groups: int,
    output_padding: Optional[Union[List[int], int]] = None,
):
    def _formula(ln: int, p: int, d: int, k: int, s: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output

        See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
        Returns:
            The output length
        """
        return (ln + 2 * p - d * (k - 1) - 1) // s + 1

    def _formula_transposed(ln: int, p: int, d: int, k: int, s: int, op: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output
        if transposed convolution is used.
        See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
            op: output padding in that dim

        Returns:
            The output length
        """
        return (ln - 1) * s - 2 * p + d * (k - 1) + op + 1

    kernel_size = weight.shape[2:]
    dims = input_tensor.shape[2:]
    if is_transposed:
        out_channels = groups * weight.shape[1]
    else:
        out_channels = weight.shape[0]
        if weight.shape[1] * groups != input_tensor.shape[1]:
            raise RuntimeError("Invalid channel dimensions")

    ret_shape = [input_tensor.shape[0], out_channels]
    if isinstance(stride, IntLike):
        stride = [stride] * len(dims)
    elif len(stride) == 1:
        stride = [stride[0]] * len(dims)

    if isinstance(padding, IntLike):
        padding = [padding] * len(dims)
    elif len(padding) == 1:
        padding = [padding[0]] * len(dims)

    if isinstance(dilation, IntLike):
        dilation = [dilation] * len(dims)
    elif len(dilation) == 1:
        dilation = [dilation[0]] * len(dims)

    output_padding_list: Optional[List[int]] = None
    if output_padding:
        if isinstance(output_padding, IntLike):
            output_padding_list = [output_padding] * len(dims)
        elif len(output_padding) == 1:
            output_padding_list = [output_padding[0]] * len(dims)
        else:
            output_padding_list = output_padding

    for i in range(len(dims)):
        # If output_padding is present, we are dealing with a transposed convolution
        if output_padding_list:
            ret_shape.append(
                _formula_transposed(
                    dims[i],
                    padding[i],
                    dilation[i],
                    kernel_size[i],
                    stride[i],
                    output_padding_list[i],
                )
            )
        else:
            ret_shape.append(
                _formula(dims[i], padding[i], dilation[i], kernel_size[i], stride[i])
            )

    return ret_shape

# quantized_decomposed_lib.define(
#     "conv2d_prepack(Tensor weight, Tensor? bias, int[] stride,"
#     "int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase")
# @impl(quantized_decomposed_lib, "conv2d_prepack", "CompositeExplicitAutograd")
# def conv2d_prepack(input, bias, stride, padding, dilation, groups):
#     # assert zero_point.numel() == 1, f"Exepecting zero_point tensor to be one element, but received : {zero_point.numel()}"
#     # assert scale.numel() == 1, f"Exepecting scale tensor to be one element, but received : {scale.numel()}"
#     # assert input.dtype == dtype, f"Expecting input to have dtype: {dtype}"
#     # if dtype in [torch.uint8, torch.int8, torch.int32]:
#     #     return torch.empty_like(input, dtype=torch.float32)
#     # else:
#     #     raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")
    
#     # return input
#     return torch.ops.quantized.conv2d_prepack(input, bias, stride, padding, dilation, groups)

# @impl(quantized_decomposed_lib, "conv2d_prepack", "Meta")
# def conv2d_prepack(input, bias, stride, padding, dilation, groups):    
#     return input
#     # return torch.ops.quantized.conv2d_prepack(input, bias, stride, padding, dilation, groups)
#     #return

# quantized_decomposed_lib.define(
#     "conv2d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase weight,"
#     "float output_scale, int output_zero_point) -> Tensor")
# @impl(quantized_decomposed_lib, "conv2d_relu", "CompositeExplicitAutograd")
# def conv2d_relu(qx, weight, output_scale, output_zero_point):
#     return torch.ops.quantized.conv2d_relu.new(qx, weight, output_scale, output_zero_point)

# @impl(quantized_decomposed_lib, "conv2d_relu", "Meta")
# def conv2d_relu(qx, weight, stride, padding, dilation, groups, output_scale, output_zero_point):
#     return torch.ops.quantized.conv2d_relu.new(qx, weight, stride, padding, dilation, groups, output_scale, output_zero_point)

quantized_decomposed_lib.define(
    "conv2d_relu.tensor(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase weight,"
    "Tensor output_scale, Tensor output_zero_point,"
    "Tensor origin_qw, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor")
@impl(quantized_decomposed_lib, "conv2d_relu.tensor", "CompositeExplicitAutograd")
def conv2d_relu(qx, weight, output_scale, output_zero_point, origin_qw, stride, padding, dilation, groups):
    print("----- hit conv2d_relu with META Dispatch Key CompositeExplicitAutograd ----", flush=True)   
    return torch.ops.quantized.conv2d_relu.new(qx, weight, output_scale, output_zero_point, origin_qw, stride, padding, dilation, groups)
@impl(quantized_decomposed_lib, "conv2d_relu.tensor", "Meta")
def conv2d_relu(qx, weight, output_scale, output_zero_point, origin_qw, stride, padding, dilation, groups):
    print("----- hit conv2d_relu with META Dispatch Key Meta ----", flush=True)
    # return torch.empty_like(qx, dtype=qx.dtype)
    print(qx.shape[2:], flush=True)
    # w, b = torch.ops.quantized.conv2d_unpack(weight)
    print(origin_qw.shape[2:], flush=True)
    shape_out = calc_conv_nd_return_shape(
        qx,
        origin_qw,
        stride,
        padding,
        dilation,
        False,
        groups,
        None,
    )
    out = qx.new_empty(shape_out)
    print("shape_out is: {}".format(shape_out))
    return out



quantized_decomposed_lib.define(
    "conv2d_relu_v2.tensor(Tensor qx, Tensor qw, Tensor? bias,"
    "int[] stride, int[] padding, int[] dilation, int groups, Tensor output_scale, Tensor output_zero_point,"
    "Tensor input_scale, Tensor input_zero_point, Tensor weight_scale, Tensor weight_zero_point, int w_axis) -> Tensor")
@impl(quantized_decomposed_lib, "conv2d_relu_v2.tensor", "CPU")
def conv2d_relu_v2(qx, qw, bias, stride, padding, dilation, groups, output_scale, output_zero_point,
    x_scale, x_zp, w_scale, w_zp, w_axis):
    print("----- hit conv2d_relu_v2 with CPU Dispatch Key ----", flush=True)
    quantized = torch.ops.quantized

    # Input Workaround to test feasibility
    # Re-constructure the qw and qx
    real_qx = torch._make_per_tensor_quantized_tensor(qx, x_scale, x_zp)
    real_qw = torch._make_per_channel_quantized_tensor(qw, w_scale, w_zp, w_axis)

    w_packed = quantized.conv2d_prepack(real_qw, bias, stride, padding, dilation, groups)
    qy = quantized.conv2d_relu.new(real_qx, w_packed, output_scale, output_zero_point)

    # Output Workround: Return a int8 tensor without quantizer
    return qy.int_repr()

@impl(quantized_decomposed_lib, "conv2d_relu_v2.tensor", "Meta")
def conv2d_relu_v2(qx, qw, bias, stride, padding, dilation, groups, output_scale, output_zero_point,
    x_scale, x_zp, w_scale, w_zp, w_axis):
    print("----- hit conv2d_relu_v2 with META Dispatch Key Meta ----", flush=True)
    #return torch.empty_like(qx, dtype=qx.dtype)
    print(qx.shape[2:], flush=True)
    # w, b = torch.ops.quantized.conv2d_unpack(weight)
    print(qw.shape[2:], flush=True)
    shape_out = calc_conv_nd_return_shape(
        qx,
        qw,
        stride,
        padding,
        dilation,
        False,
        groups,
        None,
    )
    out = qx.new_empty(shape_out)
    print("shape_out is: {}".format(shape_out))
    return out

quantized_decomposed_lib.define(
    "choose_qparams.tensor(Tensor input, int quant_min, int quant_max, "
    "ScalarType dtype) -> (Tensor, Tensor)")

@impl(quantized_decomposed_lib, "choose_qparams.tensor", "CompositeExplicitAutograd")
def choose_qparams_tensor(
        input: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert quant_min < quant_max, f"Expecting quant_min to be smaller than quant_max but received min: {quant_min} max: {quant_max}"

    # Its weird to create an observer manually just to calculate qparams. I tried refactoring this functionality out of observer
    # into a util and then use that util directly, but I kept running into jit typing errors related to torch.qscheme not
    # being recognized as a type. TODO: properly refactor this out to avoid observer overhead
    tensor_dtype_to_observer_dtype = {torch.uint8: torch.quint8, torch.int8: torch.qint8}
    observer = MinMaxObserver(quant_min=quant_min, quant_max=quant_max, dtype=tensor_dtype_to_observer_dtype[dtype])
    observer(input)
    scale, zero_point = observer.calculate_qparams()
    return (scale, zero_point)

@impl(quantized_decomposed_lib, "choose_qparams.tensor", "Meta")
def choose_qparams_tensor_meta(
        input: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert quant_min < quant_max, f"Expecting quant_min to be smaller than quant_max but received min: {quant_min} max: {quant_max}"
    return torch.empty(1, dtype=torch.float, device=input.device), torch.empty(1, dtype=torch.int32, device=input.device)

# Helper function used to implement per-channel quantization against any axis
def _permute_to_axis_zero(x, axis):
    new_axis_list = list(range(x.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(tuple(new_axis_list))
    return y, new_axis_list

quantized_decomposed_lib.define(
    "quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_channel", "CompositeExplicitAutograd")
def quantize_per_channel(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """ Affine per channel quantization for the Tensor using the same quantization
    parameters for each channel/axis to map from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 Tensor
       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel
       zero_point (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)
    res = torch.zeros_like(input)

    for i in range(input.size(0)):
        res[i] = torch.clamp(
            torch.round(input[i] * (1.0 / scales[i])) + zero_points[i],
            quant_min,
            quant_max
        )

    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)

@impl(quantized_decomposed_lib, "quantize_per_channel", "Meta")
def quantize_per_channel_meta(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)

# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_per_channel", "CompositeExplicitAutograd")
def dequantize_per_channel(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """ Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per channel quantized Tensor if combined with
       quantization parameter in the argument of this function (scales/zero_points/axis)

       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel

       zero_points (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel

       quant_min (int): minimum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): requested dtype for output Tensor (not used in computation,
       reserved for pattern matching)

    Returns:
       dequantized float32 Tensor
    """
    assert input.dtype == dtype, f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)
    res = torch.zeros_like(input, dtype=torch.float32)

    for i in range(input.size(0)):
        # TODO: investigate why
        # (input[i] - zero_points[i]).to(torch.float32) * scales[i]
        # failed the test
        res[i] = (input[i].to(torch.float32) - zero_points[i]) * scales[i]

    out = res.permute(tuple(permute_axis_list))
    return out

@impl(quantized_decomposed_lib, "dequantize_per_channel", "Meta")
def dequantize_per_channel_meta(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    assert input.dtype == dtype, f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=torch.float32)
