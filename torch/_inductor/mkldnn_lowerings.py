from typing import List, Optional

import torch
import torch.utils._pytree as pytree
from torch._inductor.kernel.mm_common import mm_args
from . import ir
from .codegen.cpp_gemm_template import CppPackedGemmTemplate
from .ir import TensorBox
from .lowering import (
    add,
    add_needs_realized_inputs,
    aten,
    permute,
    register_lowering,
    to_dtype,
    view,
)
from .select_algorithm import (
    autotune_select_algorithm,
    ChoiceCaller,
    ExternKernelChoice,
)
from .utils import use_aten_gemm_kernels, use_cpp_packed_gemm_template, use_max_autotune
from .virtualized import V


def register_onednn_fusion_ops():
    if torch._C._has_mkldnn:
        aten_mkldnn_linear_unary = ExternKernelChoice(
            torch.ops.mkldnn._linear_pointwise,
            "mkldnn::_linear_pointwise",
            has_out_variant=False,
            kernel_creator=ir.LinearUnary.create,
        )
        aten_mkldnn_qlinear_unary = ExternKernelChoice(
            torch.ops.onednn.qlinear_pointwise,
            "onednn::qlinear_pointwise",
            has_out_variant=False,
            kernel_creator=ir.QLinearPointwisePT2E.create,
        )
        cpu_needs_realized_inputs = [
            torch.ops.mkldnn._convolution_pointwise,
            torch.ops.mkldnn._convolution_pointwise_,
            torch.ops.mkldnn._convolution_transpose_pointwise,
            torch.ops.mkldnn._linear_pointwise,
            aten.mkldnn_rnn_layer.default,
            torch.ops.onednn.qconv2d_pointwise,
        ]

        @register_lowering(torch.ops.mkldnn._convolution_pointwise)
        def convolution_unary(
            x: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            attr,
            scalars,
            algorithm,
        ):
            return TensorBox.create(
                ir.ConvolutionUnary.create(
                    x,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._convolution_pointwise.binary)
        def convolution_binary(
            x: TensorBox,
            other: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            return TensorBox.create(
                ir.ConvolutionBinary.create(
                    x,
                    other,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    binary_attr,
                    binary_alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._convolution_pointwise_.binary)
        def convolution_binary_inplace(
            x: TensorBox,
            other: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            return TensorBox.create(
                ir.ConvolutionBinaryInplace.create(
                    x,
                    other,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    binary_attr,
                    binary_alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._linear_pointwise)
        def linear_unary(
            x: TensorBox,
            w: TensorBox,
            b: TensorBox,
            attr,
            scalars,
            algorithm,
            layout=None,
        ):
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            choices: List[ChoiceCaller] = []
            if len(choices) == 0 or use_aten_gemm_kernels():
                choices.append(
                    aten_mkldnn_linear_unary.bind(
                        (x, w),
                        layout,
                        B=None,
                        attr=attr,
                        scalars=scalars,
                        algorithm=algorithm,
                    )
                    if b is None
                    else aten_mkldnn_linear_unary.bind(
                        (x, w, b),
                        layout,
                        attr=attr,
                        scalars=scalars,
                        algorithm=algorithm,
                    )
                )
            if use_max_autotune():
                transposed_w = permute(w, [1, 0])
                *_, layout, x, transposed_w = mm_args(x, transposed_w, layout=layout)
                # TODO(jgong5): support epilogue fusion
                if (
                    use_cpp_packed_gemm_template(layout, x, transposed_w)
                    and attr == "none"
                ):
                    if b is None:
                        CppPackedGemmTemplate.add_choices(
                            choices,
                            layout,
                            [x, w],
                            trans_w=True,
                        )
                    else:
                        CppPackedGemmTemplate.add_choices(
                            choices,
                            layout,
                            [x, w, b],
                            trans_w=True,
                            input_indices=[2, 0, 1],
                        )
            assert w.get_name() in V.graph.constants
            input_gen_fns = {
                1: lambda x: V.graph.constants[x.get_name()],
            }
            result = autotune_select_algorithm(
                "linear_unary",
                choices,
                [x, w] if b is None else [x, w, b],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

        @register_lowering(torch.ops.mkldnn._linear_pointwise.binary)
        def linear_binary(x: TensorBox, y: TensorBox, w: TensorBox, b: TensorBox, attr):
            return TensorBox.create(ir.LinearBinary.create(x, y, w, b, attr))

        @register_lowering(torch.ops.mkldnn._convolution_transpose_pointwise)
        def convolution_transpose_unary(
            x: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            attr,
            scalars,
            algorithm,
        ):
            return TensorBox.create(
                ir.ConvolutionTransposeUnary.create(
                    x,
                    weight,
                    bias,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(aten.mkldnn_rnn_layer.default)
        def mkldnn_rnn_layer(
            x: TensorBox,
            w0: TensorBox,
            w1: TensorBox,
            w2: TensorBox,
            w3: TensorBox,
            hx: TensorBox,
            cx: TensorBox,
            reverse: bool,
            batch_sizes: List[int],
            mode: int,
            hidden_size: int,
            num_layers: int,
            has_biases: bool,
            bidirectional: bool,
            batch_first: bool,
            train: bool,
        ):
            return pytree.tree_map(
                TensorBox.create,
                ir.MkldnnRnnLayer.create(
                    x,
                    w0,
                    w1,
                    w2,
                    w3,
                    hx,
                    cx,
                    reverse,
                    batch_sizes,
                    mode,
                    hidden_size,
                    num_layers,
                    has_biases,
                    bidirectional,
                    batch_first,
                    train,
                ),
            )

        @register_lowering(torch.ops.onednn.qconv2d_pointwise, type_promotion_kind=None)
        def qconvolution_unary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp: TensorBox,
            bias: TensorBox,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            attr,
            scalars,
            algorithm,
        ):
            return TensorBox.create(
                ir.QConvPointWisePT2E.create(
                    x,
                    x_scale,
                    x_zp,
                    packed_weight,
                    w_scale,
                    w_zp,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(
            torch.ops.onednn.qconv2d_pointwise.binary, type_promotion_kind=None
        )
        def qconvolution_binary(
            x: TensorBox,
            x_scale,
            x_zp,
            accum: TensorBox,
            accum_scale,
            accum_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp: TensorBox,
            bias: TensorBox,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithmm,
        ):
            if (
                binary_attr == "sum"
                and output_dtype in [torch.float32, torch.bfloat16]
                and accum.get_dtype() in [torch.float32, torch.bfloat16]
                and accum.get_dtype() != output_dtype
            ):
                # For int8-mixed-bf16 quantization and inplace add,
                # there is case when accum dtype is float32 but output dtype is bfloat16.
                # Since the accum will be inplaced changed with post op sum,
                # we will do accum dtype convertion here.
                accum = to_dtype(accum, output_dtype)
            return TensorBox.create(
                ir.QConvPointWiseBinaryPT2E.create(
                    x,
                    x_scale,
                    x_zp,
                    accum,
                    accum_scale,
                    accum_zp,
                    packed_weight,
                    w_scale,
                    w_zp,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    binary_attr,
                    alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithmm,
                )
            )

        @register_lowering(torch.ops.onednn.qlinear_pointwise, type_promotion_kind=None)
        def qlinear_unary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp: TensorBox,
            bias: TensorBox,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            attr,
            scalars,
            algorithm,
            layout=None,
        ):
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            print("type(x_scale) is: {}".format(type(x_scale)), flush=True)
            print("type(x_zp) is: {}".format(type(x_zp)), flush=True)
            print("type(w_scale) is: {}".format(type(w_scale)), flush=True)

            if not isinstance(x_scale, ir.TensorBox):
                assert type(x_scale) == float
                x_scale = V.graph.add_tensor_constant(torch.tensor(x_scale, dtype=torch.float32), name="x_scale")
            if not isinstance(x_zp, ir.TensorBox):
                assert type(x_zp) == int
                x_zp = V.graph.add_tensor_constant(torch.tensor(x_zp, dtype=torch.int32), name="x_zp")
            # breakpoint()
            if w_zp.get_dtype() != torch.int32:
                w_zp_tensor = V.graph.constants[w_zp.get_name()].to(torch.int32)
                w_zp = V.graph.add_tensor_constant(torch.tensor(w_zp_tensor, dtype=torch.int32), name=w_zp.get_name())

            choices: List[ChoiceCaller] = []
            if len(choices) == 0 or use_aten_gemm_kernels():
                # print("---- w_zp is: {}".format(w_zp), flush=True)
                # print(V.graph.constant_name(w_zp.get_name(), device_override=None), flush=True)
                # breakpoint()
                choices.append(
                    aten_mkldnn_qlinear_unary.bind(
                        (x, x_scale, x_zp, packed_weight, w_scale, w_zp),
                        layout,
                        bias=None,
                        output_scale=o_inv_scale,
                        output_zero_point=o_zero_point,
                        output_dtype=output_dtype,
                        post_op_name=attr,
                        post_op_args=scalars,
                        post_op_algorithm=algorithm,
                    )
                    if bias is None
                    else aten_mkldnn_qlinear_unary.bind(
                        (x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias),
                        layout,
                        output_scale=o_inv_scale,
                        output_zero_point=o_zero_point,
                        output_dtype=output_dtype,
                        post_op_name=attr,
                        post_op_args=scalars,
                        post_op_algorithm=algorithm,
                    )
                )
            if use_max_autotune():
                # transposed_w = permute(packed_weight, [1, 0])
                transposed_w = packed_weight
                # Only support u8s8f32 for now
                assert output_dtype == torch.float32
                *_, layout, x, transposed_w = mm_args(x, transposed_w, layout=layout, out_dtype=output_dtype)
                # TODO(jgong5): support epilogue fusion
                print("---- start to check use_cpp_packed_gemm_template ----", flush=True)
                print("torch.equal(torch.zeros_like(V.graph.constants[w_zp.get_name()]), V.graph.constants[w_zp.get_name()]) is: {}".format(
                    torch.equal(torch.zeros_like(V.graph.constants[w_zp.get_name()]), V.graph.constants[w_zp.get_name()])
                ), flush=True)
                if (
                    use_cpp_packed_gemm_template(layout, x, transposed_w, input_dtype=torch.uint8)
                    and attr == "none"
                    # We only composentate MatrixB and assume B_zp is 0 to avoid the composantation of MatrixA
                    and torch.equal(torch.zeros_like(V.graph.constants[w_zp.get_name()]), V.graph.constants[w_zp.get_name()])
                ):
                    # print("----- use_cpp_packed_gemm_template is: {}".format(use_cpp_packed_gemm_template(layout, x, transposed_w)), flush=True)
                    print("----- hit the int8 gemm templare ----", flush=True)
                    print(bias is None, flush=True)
                    if bias is None:
                        print("---- x.get_size() is: {}".format(x.get_size()), flush=True)
                        print("---- packed_weight.get_size() is: {}".format(packed_weight.get_size()), flush=True)
                        CppPackedGemmTemplate.add_choices(
                            choices,
                            layout,
                            [x, x_scale, x_zp, permute(packed_weight, [1, 0]), w_scale, w_zp],
                            trans_w=True,
                            input_dtype=torch.uint8,
                        )
                    else:
                        assert bias is None
                    #     CppPackedGemmTemplate.add_choices(
                    #         choices,
                    #         layout,
                    #         [x, packed_weight, bias],
                    #         trans_w=True,
                    #         input_indices=[2, 0, 1],
                    #     )
            assert packed_weight.get_name() in V.graph.constants
            input_gen_fns = {
                1: lambda x: V.graph.constants[x.get_name()],
                2: lambda x: V.graph.constants[x.get_name()],
                3: lambda x: V.graph.constants[x.get_name()],
                4: lambda x: V.graph.constants[x.get_name()],
                5: lambda x: V.graph.constants[x.get_name()],
            }
            result = autotune_select_algorithm(
                "qlinear_unary",
                choices,
                [x, x_scale, x_zp, packed_weight, w_scale, w_zp] if bias is None else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result
            # return TensorBox.create(
            #     ir.QLinearPointwisePT2E.create(
            #         x,
            #         x_scale,
            #         x_zp,
            #         packed_weight,
            #         w_scale,
            #         w_zp,
            #         bias,
            #         o_inv_scale,
            #         o_zero_point,
            #         output_dtype,
            #         attr,
            #         scalars,
            #         algorithm,
            #     )
            # )

        if torch._C.has_mkl:
            aten_mkl_linear = ExternKernelChoice(
                torch.ops.mkl._mkl_linear,
                "mkl::_mkl_linear",
                has_out_variant=False,
                kernel_creator=ir.MKLPackedLinear.create,
            )
            cpu_needs_realized_inputs.append(torch.ops.mkl._mkl_linear)

            @register_lowering(torch.ops.mkl._mkl_linear)
            def mkl_packed_linear(
                x: TensorBox,
                packed_w: TensorBox,
                orig_w: TensorBox,
                b: Optional[TensorBox],
                batch_size,
                *,
                layout=None,
            ):
                choices: List[ChoiceCaller] = []
                if use_max_autotune():
                    transposed_w = permute(orig_w, [1, 0])
                    *_, layout, x, transposed_w = mm_args(
                        x, transposed_w, layout=layout
                    )
                    if use_cpp_packed_gemm_template(layout, x, transposed_w):
                        CppPackedGemmTemplate.add_choices(
                            choices,
                            layout,
                            [x, packed_w, orig_w],
                            trans_w=True,
                            input_indices=[0, 2],
                        )

                if len(choices) == 0 or use_aten_gemm_kernels():
                    choices.append(
                        aten_mkl_linear.bind(
                            (x, packed_w, orig_w), layout, B=None, batch_size=batch_size
                        )
                    )

                assert packed_w.get_name() in V.graph.constants
                assert orig_w.get_name() in V.graph.constants
                # packed_w is a mkldnn tensor which we can't generate directly
                # so we use the weights from the original tensor in autotune.
                input_gen_fns = {
                    1: lambda x: V.graph.constants[x.get_name()],
                    2: lambda x: V.graph.constants[x.get_name()],
                }
                result: TensorBox = autotune_select_algorithm(
                    "packed_linear",
                    choices,
                    [x, packed_w, orig_w],
                    layout,
                    input_gen_fns=input_gen_fns,
                )
                if b is not None:
                    result = add(result, b)
                return result

        add_needs_realized_inputs(cpu_needs_realized_inputs)
    else:
        pass
