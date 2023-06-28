import torch
from ..ir import QConv, IPEXQConv
from ..pattern_matcher import Arg, CallFunction, KeywordArg, Match, register_graph_pattern
from .post_grad import register_lowering_pattern, pass_patterns
import copy

aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed
dequantize_per_channel = quantized_decomposed.dequantize_per_channel.default

"""
dequantize activation:
    x = x.to(fp32)
    x = x - zero_point
    x = x * scale
"""
dequantize_activation_pattern = CallFunction(
    aten.mul.Tensor,
    CallFunction(
        aten.sub.Tensor,
        CallFunction(
            prims.convert_element_type.default,
            KeywordArg("x"),
            KeywordArg("x_dq_dtype"),
        ),
        KeywordArg("x_zp"),
    ),
    KeywordArg("x_scale"),
)

dequantize_weight_pattern = CallFunction(
    dequantize_per_channel,
    KeywordArg("w"),
    KeywordArg("w_scale"),
    KeywordArg("w_zp"),
    KeywordArg("w_axis"),  # axis for quantization
    KeywordArg("w_qmin"),  # quant clamp min
    KeywordArg("w_qmax"),  # quant clamp max
    KeywordArg("qw_dtype"),  # dtype=torch.int8
)

aten_conv_pattern = CallFunction(
    aten.convolution.default,
    dequantize_activation_pattern,
    dequantize_weight_pattern,
    KeywordArg("b"),  # bias
    KeywordArg("stride"),
    KeywordArg("padding"),
    KeywordArg("dilation"),
    KeywordArg("transposed"),
    KeywordArg("o_padding"),
    KeywordArg("groups"),
)

"""
quantize output:
    scale = 1 / scale
    scale = 1.0 * scale
    output = round(output * scale)
    output = output + zero_point
    output = clamp_min(output, 0)
    output = clamp_max(output, 127)
    output = output.to(uint8)
"""
quantize_conv_output_pattern = CallFunction(
    prims.convert_element_type.default,
    CallFunction(
        aten.clamp_max.default,
        CallFunction(
            aten.clamp_min.default,
            CallFunction(
                aten.add.Tensor,
                CallFunction(
                    aten.round.default,
                    CallFunction(
                        aten.mul.Tensor,
                        aten_conv_pattern,  # output of conv
                        CallFunction(
                            aten.mul.Tensor,
                            CallFunction(
                                aten.reciprocal.default, KeywordArg("o_scale")
                            ),
                            Arg(),  # 1.0
                        ),
                    ),
                ),
                KeywordArg("o_zp"),
            ),
            KeywordArg("o_qmin"),  # 0
        ),
        KeywordArg("o_qmax"),  # 127
    ),
    KeywordArg("o_dtype"),  # dtype=torch.uint8
)


def _register_quantized_conv_lowering(pattern):
    @register_lowering_pattern(pattern)
    def qconv(match: Match, *args, **kwargs):
        x, x_scale, x_zp = kwargs["x"], kwargs["x_scale"], kwargs["x_zp"]
        w, w_scale, w_zp, w_axis = (
            kwargs["w"],
            kwargs["w_scale"],
            kwargs["w_zp"],
            kwargs["w_axis"],
        )
        b, stride, padding, dilation = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
        )
        groups, o_scale, o_zero_point, o_dtype = (
            kwargs["groups"],
            kwargs["o_scale"],
            kwargs["o_zp"],
            kwargs["o_dtype"],
        )
        weight_shape = w.get_size()
        dim = len(weight_shape) - 2
        return QConv.create(
            dim,
            x,
            x_scale,
            x_zp,
            w,
            w_scale,
            w_zp,
            w_axis,
            b,
            stride,
            padding,
            dilation,
            groups,
            o_scale,
            o_zero_point,
            o_dtype,
        )

    return qconv

ipex_aten_conv_pattern = CallFunction(
    torch.ops.torch_ipex.prepacked_dynamic_conv.tensor,
    dequantize_activation_pattern,
    dequantize_weight_pattern,
    KeywordArg("b"),  # bias
    KeywordArg("stride"),
    KeywordArg("padding"),
    KeywordArg("dilation"),
    KeywordArg("transposed"),
    KeywordArg("o_padding"),
    KeywordArg("groups"),
    KeywordArg("packed_weight"),
)

quantize_ipex_conv_output_pattern = CallFunction(
    prims.convert_element_type.default,
    CallFunction(
        aten.clamp_max.default,
        CallFunction(
            aten.clamp_min.default,
            CallFunction(
                aten.add.Tensor,
                CallFunction(
                    aten.round.default,
                    CallFunction(
                        aten.mul.Tensor,
                        ipex_aten_conv_pattern,  # output of conv
                        KeywordArg("o_inv_scale"),
                    ),
                ),
                KeywordArg("o_zp"),
            ),
            KeywordArg("o_qmin"),  # 0
        ),
        KeywordArg("o_qmax"),  # 127
    ),
    KeywordArg("o_dtype"),  # dtype=torch.uint8
)

pattern_match_count = 0
def _register_ipex_quantized_conv_lowering(pattern):
    @register_lowering_pattern(pattern)
    def qconv(match: Match, *args, **kwargs):
        x, x_scale, x_zp = kwargs["x"], kwargs["x_scale"], kwargs["x_zp"]
        w, w_scale, w_zp, w_axis = (
            kwargs["w"],
            kwargs["w_scale"],
            kwargs["w_zp"],
            kwargs["w_axis"],
        )
        b, stride, padding, dilation = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
        )
        groups, o_inv_scale, o_zero_point, o_dtype = (
            kwargs["groups"],
            kwargs["o_inv_scale"],
            kwargs["o_zp"],
            kwargs["o_dtype"],
        )

        packed_weight = kwargs["packed_weight"]

        global pattern_match_count
        pattern_match_count += 1
        print("---- matched the pattern ----: {}".format(pattern_match_count), flush=True)

        weight_shape = w.get_size()
        dim = len(weight_shape) - 2
        return IPEXQConv.create(
            dim,
            x,
            x_scale,
            x_zp,
            w,
            w_scale,
            w_zp,
            w_axis,
            b,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            o_dtype,
            packed_weight,
        )

    return qconv

dequant_node_pattern = CallFunction(
    aten.mul.Tensor,
    CallFunction(
        aten.sub.Tensor,
        CallFunction(
            prims.convert_element_type.default,
            KeywordArg("x"),
            KeywordArg("o_dtype"),  # dtype=torch.float32 
        ),
        KeywordArg("dequant_zp"),  # dequant zp
    ),
    KeywordArg("dequant_scale"),  # dequant_scale
)
def _register_dequant_promotion_pass(pattern):
    @register_graph_pattern(pattern, pass_dict=pass_patterns[0],) #pass_number=0, so it will run before quantizatioin fusion
    def dequant_promotion(match: Match, *args, **kwargs):
        # print("---- find hit the dequant pattern -----", flush=True)
        to_fp32_node = match.nodes[0]
        sub_node = match.nodes[1]
        mul_node = match.nodes[2]
        graph = match.graph
        # print("to_fp32_node is: {}".format(to_fp32_node), flush=True)
        # print("sub_node is: {}".format(sub_node), flush=True)
        # print("mul_node is: {}".format(mul_node), flush=True)
        if len(list(mul_node.users)) > 1:
            # Dequant Node used by multiply nodes
            # Will do dequant promotion, so each used node has a seperate dequant pattern connected
            for index in range(len(list(mul_node.users)) -1):
                user_node = list(mul_node.users)[index]
                with graph.inserting_before(user_node):
                    # Step1: Duplicate the mul node
                    new_mul_node = graph.call_function(
                        torch.ops.aten.mul.Tensor,
                        args=mul_node.args,
                        kwargs=mul_node.kwargs,
                    )
                    new_mul_node.meta = copy.copy(mul_node.meta)
                    user_node.replace_input_with(mul_node, new_mul_node)

                    with graph.inserting_before(new_mul_node):
                        # Step2: Duplicate the sub node
                        new_sub_node = graph.call_function(
                            torch.ops.aten.sub.Tensor,
                            args=sub_node.args,
                            kwargs=sub_node.kwargs,
                        )
                        new_sub_node.meta = copy.copy(sub_node.meta)
                        new_mul_node.replace_input_with(sub_node, new_sub_node)

                        with graph.inserting_before(new_sub_node):
                            # Step3: Duplicate the to_fp32 node
                            new_to_fp32_node = graph.call_function(
                                torch.ops.prims.convert_element_type.default,
                                args=to_fp32_node.args,
                                kwargs=to_fp32_node.kwargs,
                            )
                            new_to_fp32_node.meta = copy.copy(to_fp32_node.meta)
                            new_sub_node.replace_input_with(to_fp32_node, new_to_fp32_node)

def register_quantization_lowerings():
    _register_quantized_conv_lowering(quantize_conv_output_pattern)
    _register_ipex_quantized_conv_lowering(quantize_ipex_conv_output_pattern)
    _register_dequant_promotion_pass(dequant_node_pattern)
