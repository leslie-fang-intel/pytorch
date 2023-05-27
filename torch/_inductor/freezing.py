import itertools

import unittest
import weakref
from typing import List, Optional, Tuple

import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree

from torch._dynamo.utils import detect_fake_mode
from torch._functorch.compile_utils import fx_graph_cse
from torch._inductor.fx_passes.freezing_patterns import get_freezing_patterns
from torch.ao.quantization._pt2e.utils import _fuse_conv_bn_
from torch.fx.experimental.proxy_tensor import make_fx
from . import config
from .decomposition import select_decomp_table
import torch.ao.quantization.fx._decomposed

aten = torch.ops.aten


def replace_node_with_constant(gm, node, constant):
    g = gm.graph

    i = 0
    while True:
        qualname = f"_frozen_param{i}"
        if not hasattr(gm, qualname):
            break
        i += 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


def replace_params_with_constants(gm, flat_params, fw_metadata) -> List[int]:
    """
    Replaces the parameters of a PyTorch GraphModule with constants wherever possible.

    Returns a list of indices representing the input parameters that were not converted to constants.
    """

    params = [node for node in gm.graph.nodes if node.op == "placeholder"]
    fake_inp_nodes = [node for (_, node) in zip(flat_params, params)]

    g = gm.graph

    preserved_arg_indices = []
    aliased_input_args = [
        out_info.base_idx
        for out_info in fw_metadata.output_info
        if out_info.base_idx is not None
    ]

    for i, (real_input, node) in enumerate(zip(flat_params, fake_inp_nodes)):
        if i in fw_metadata.mutated_inp_indices or aliased_input_args:
            preserved_arg_indices.append(i)
            continue

        replace_node_with_constant(gm, node, real_input)

    # add on non param inputs
    preserved_arg_indices.extend(range(len(flat_params), len(params)))

    # is this necessary ?
    gm.recompile()
    return preserved_arg_indices


@torch.utils._python_dispatch._disable_current_modes()
def constant_fold(gm):
    unknown_value = object()

    node_replacements = {}

    class ConstantFolder(torch.fx.Interpreter):
        def run_node(self, node):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            if unknown_value in pytree.tree_flatten((args, kwargs))[0]:
                return unknown_value

            # All mutations should either be removed or on inputs which we did not make constant
            if (
                isinstance(node.target, torch._ops.OpOverload)
                and torch.Tag.nondeterministic_seeded in node.target.tags
            ):
                return unknown_value

            out = super().run_node(node)

            # TODO - remove constant from node_replacement when it has no uses

            # print("node.target is: {}".format(node.target), flush=True)

            if node.op != "get_attr" and isinstance(out, torch.Tensor):
                if node.target == torch.ops.quantized_decomposed.dequantize_per_channel.default:
                # if isinstance(node.target, torch.ops.quantized_decomposed):
                    return out
                node_replacements[node] = out

            return out

        def run(self):
            env = {}
            for n in self.module.graph.nodes:
                if n.op == "placeholder":
                    env[n] = unknown_value
            return super().run(initial_env=env)

    ConstantFolder(gm).run()

    for node, constant in node_replacements.items():
        replace_node_with_constant(gm, node, constant)

    erased_params = []
    for node in gm.graph.nodes:
        if node.op == "get_attr" and len(node.users) == 0:
            delattr(gm, node.target)
            erased_params.append(node)

    for node in erased_params:
        gm.graph.erase_node(node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


@torch.utils._python_dispatch._disable_current_modes()
def fuse_conv_bn(gm):
    return _fuse_conv_bn_(gm)

@torch.utils._python_dispatch._disable_current_modes()
def quantization_weight_prepack(gm):
    print(" inside quantization_weight_prepack gm is : {}".format(gm), flush=True)
    # Assume dq - conv (- relu) - q is already fused and `w - q` is replaced by qw
    decomposed = torch.ops.quantized_decomposed
    for node in gm.graph.nodes:
        print("node.target is: {}".format(node.target), flush=True)
        if node.target == aten.convolution.default:
            conv_node = node
            (
                x,
                w,
                bias,
                stride,
                padding,
                dilation,
                is_transposed,
                out_padding,
                groups,
            ) = conv_node.args
            # assert (
            #     x.target == dequantize_per_tensor
            # ), "input's node should be dequantize_per_tensor"
            assert (
                w.target == decomposed.dequantize_per_channel.default
            ), "weight's node should be dequantize_per_channel"
            # (qx, x_scale, x_zp, x_quant_min, x_quant_max, x_dtype) = x.args
            (qw, w_scale, w_zp, w_axis, w_quant_min, w_quant_max, w_dtype) = w.args
            assert (
                x.target == torch.ops.aten.mul.Tensor
            ), "input's node should be dequantize_per_tensor"
            mul = x
            (
                sub,
                x_scale,
            ) = mul.args       
            assert (
                sub.target == torch.ops.aten.sub.Tensor
            ), "input's node should be dequantize_per_tensor"
            (
                to_fp32,
                x_zp,
            ) = sub.args
            assert (
                to_fp32.target == torch.ops.prims.convert_element_type.default
            ), "input's node should be dequantize_per_tensor"
            (
                qx,
                _,
            ) = to_fp32.args
            x_shape = qx.meta.get("tensor_meta").shape
            print("x_shape is: {}".format(x_shape), flush=True)
            
            weight_int8_tensor = getattr(gm, qw.target)
            bias_tensor = getattr(gm, bias.target) if bias is not None else None
            w_scale_tensor = getattr(gm, w_scale.target)
            x_scale_tensor = getattr(gm, x_scale.target)
            x_zp_tensor = getattr(gm, x_zp.target)

            packed_weight, packed_bias = torch.ops.quantized.conv_prepack_cpu_tensor(
                weight_int8_tensor,
                w_scale_tensor,
                x_shape,
                x_scale_tensor,
                x_zp_tensor,
                bias_tensor,
                stride,
                padding,
                dilation,
                groups,
            )

            w_attr_name = qw.target
            w_packed_attr_name = w_attr_name + "_packed"
            gm.graph.owning_module._buffers[w_packed_attr_name] = packed_weight
            setattr(gm, w_packed_attr_name, gm.graph.owning_module._buffers[w_packed_attr_name])
            # qw.target = w_packed_attr_name
            with gm.graph.inserting_before(qw):
                prepack_weight_node = gm.graph.get_attr(w_packed_attr_name)
                #prepack_weight_node.meta.update(qw.meta)

            if bias is not None:
                b_attr_name = bias.target
                b_pack_attr_name = b_attr_name + "_packed"
                gm.graph.owning_module._buffers[b_pack_attr_name] = packed_bias
                setattr(gm, b_pack_attr_name, gm.graph.owning_module._buffers[b_pack_attr_name])
                with gm.graph.inserting_before(qw):
                    prepack_bias_node = gm.graph.get_attr(b_pack_attr_name)
            else:
                prepack_bias_node = bias
                # bias_node.target = b_pack_attr_name
                # delattr(gm, b_attr_name)

            # print("packed_weight is: {}".format(packed_weight), flush=True)
            # print("packed_bias is: {}".format(packed_bias), flush=True)

            with gm.graph.inserting_after(conv_node):
                new_args = (
                    x,
                    w,
                    bias,
                    stride,
                    padding,
                    dilation,
                    is_transposed,
                    out_padding,
                    groups,
                    prepack_weight_node,
                    prepack_bias_node,
                )
                new_conv_node = gm.graph.call_function(
                    torch.ops.quantized.prepacked_dynamic_conv.tensor, args=new_args
                )
                conv_node.replace_all_uses_with(new_conv_node)
                new_conv_node.meta.update(conv_node.meta)
                gm.graph.erase_node(conv_node)
            # conv_args = list(conv_node.args)
            # conv_args[1] = qw
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm

def decompose_unfused_batchnorms(gm, example_inputs, preserved_arg_indices):
    if not any(
        node.target is aten._native_batch_norm_legit_no_training.default
        for node in gm.graph.nodes
    ):
        return gm

    fake_mode = detect_fake_mode(example_inputs)

    # constant params will be real tensors, not fake
    # TODO: fake_mode should should enable py dispatcher if its symbolic ?
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
        args = [e for i, e in enumerate(example_inputs) if i in preserved_arg_indices]
        with fx_traceback.preserve_node_meta():
            gm = make_fx(gm, select_decomp_table())(*args)

    return gm


def freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: List[torch._subclasses.FakeTensor],
) -> Tuple[torch.fx.GraphModule, List[int]]:
    """
    Inlines parameters that are not mutated into constants and optimizes the graph through constant propagation
    and other techniques. If enabled, the function also discards the original parameters of the module for memory efficiency.

    Assumes that this function is run in dynamo tracing post aot_autograd.

    Args:
        dynamo_gm (torch.fx.GraphModule): The Dynamo constructed GraphModule.
        aot_autograd_gm (torch.fx.GraphModule): The aot_autograd constructed GraphModule to be frozen.
        example_inputs (List[torch.Tensor]): A list of example input tensors to be used in the freezing process.

    Returns:
        Tuple[torch.fx.GraphModule, List[int]]: A tuple containing the frozen GraphModule and a list of indices
        of the inputs that were preserved (not turned into constants).
    """
    fw_metadata = torch._guards.TracingContext.get().fw_metadata
    params_flat = torch._guards.TracingContext.get().params_flat
    assert fw_metadata is not None and params_flat is not None

    preserved_arg_indices = replace_params_with_constants(
        aot_autograd_gm, params_flat, fw_metadata
    )

    constant_fold(aot_autograd_gm)

    fuse_conv_bn(aot_autograd_gm)

    quantization_weight_prepack(aot_autograd_gm)

    # now, decomp batch norm if we were unable to fuse it
    aot_autograd_gm = decompose_unfused_batchnorms(
        aot_autograd_gm, example_inputs, preserved_arg_indices
    )

    # TODO - further restrict cse ? right now needed to dedup aliasing ops
    cse_graph = fx_graph_cse(aot_autograd_gm.graph)
    aot_autograd_gm.graph = cse_graph
    aot_autograd_gm.recompile()

    patterns = get_freezing_patterns()

    patterns.apply(aot_autograd_gm.graph)

    # TODO - apply legalization in pattern matcher
    torch.fx.passes.tools_common.legalize_graph(aot_autograd_gm)
    constant_fold(aot_autograd_gm)

    # invalidate nn Modules
    if config.freezing_discard_parameters:
        invalidate_eager_modules()
        discard_traced_gm_params(dynamo_gm)
    return aot_autograd_gm, preserved_arg_indices


class ErasedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, name, owning_mod):
        return super().__new__(cls, elem.to(device="meta"))

    def __init__(self, elem, name: Optional[str], mod):
        self.erased_name = name
        self.owning_mod_ref = weakref.ref(mod)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        erased_tensors = [
            e
            for e in pytree.tree_flatten((args, kwargs))[0]
            if isinstance(e, ErasedTensor)
        ]
        assert len(erased_tensors) > 0
        e = erased_tensors[0]

        raise RuntimeError(
            f"Trying to Run Pytorch Eager Module After Dynamo Freezing. "
            "The original parameters have been discarded for memeory efficiency. "
            f"Found in op {func} for erased parameter {e.erased_name} of {e.owning_mod_ref()}"
        )


@torch.utils._python_dispatch._disable_current_modes()
def invalidate_eager_modules():
    for mod in torch._guards.TracingContext.get().module_context.nn_modules.values():
        if not isinstance(mod, torch.nn.Module):
            continue

        for attr_name, tensor in list(
            itertools.chain(
                mod.named_parameters(recurse=False), mod.named_buffers(recurse=False)
            )
        ):
            e_t = ErasedTensor(tensor, attr_name, mod)
            if isinstance(tensor, torch.nn.Parameter):
                e_t.requires_grad_(True)
                e_t._is_param = True
            setattr(mod, attr_name, e_t)


@torch.utils._python_dispatch._disable_current_modes()
def discard_traced_gm_params(mod):
    for attr_name, tensor in list(
        itertools.chain(
            mod.named_parameters(recurse=False), mod.named_buffers(recurse=False)
        )
    ):
        e_t = ErasedTensor(tensor, attr_name, mod)
        if isinstance(tensor, torch.nn.Parameter):
            e_t.requires_grad_(True)
            e_t._is_param = True
        setattr(mod, attr_name, e_t)
