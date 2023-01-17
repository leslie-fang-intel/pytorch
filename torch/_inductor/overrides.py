import copy
import logging
import random
import weakref

import torch
import torch.nn as nn
from torch import _prims
from torch._dynamo.utils import fake_mode_from_tensors
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights
from torch.overrides import TorchFunctionMode
from torch.fx import subgraph_rewriter

from . import config
from .fx_utils import matches_module_function_pattern

from .mkldnn import mkldnn_fuse_fx

log = logging.getLogger(__name__)


class AutogradMonkeypatch(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if not kwargs:
            kwargs = {}
        if func in replacements and not (
            config.fallback_random
            and replacements[func] in replacements_using_triton_random
        ):
            return replacements[func](*args, **kwargs)
        return func(*args, **kwargs)


patch_functions = AutogradMonkeypatch


def replace_fx(gm: torch.fx.GraphModule):
    # Sometimes patch_functions() misses things already in the graph
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "call_function" and node.target in replacements:
            if (
                config.fallback_random
                and replacements[node.target] in replacements_using_triton_random
            ):
                continue
            with gm.graph.inserting_before(node):
                node.replace_all_uses_with(
                    gm.graph.call_function(
                        replacements[node.target], node.args, node.kwargs
                    )
                )
            gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm


def fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    is_cpu = all(
        example_input.device == torch.device("cpu") for example_input in example_inputs
    )

    fake_mode = fake_mode_from_tensors(example_inputs)

    gm = sink_cat_after_pointwise(gm)
    if config.permute_fusion and not is_cpu:
        # For linear permute fusion, we need to check input info to identify
        # and perform proper permutation/transpose
        ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)
        gm = linear_permute_fusion(gm)
        gm = permute_linear_fusion(gm)
        gm = permute_matmul_fusion(gm)

    # make sure the autograd is disabled.
    if torch.is_grad_enabled():
        return gm
    if not is_cpu:
        return gm
    gm = remove_identity(gm)
    gm = fuse_conv_bn(gm)
    # do mkldnn fusion(conv(linear)+unary(binary)
    gm = mkldnn_fuse_fx(gm, example_inputs)
    return gm


def fetch_attr(target: str, mod):
    target_atoms = target.split(".")
    attr_itr = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def remove_identity(gm: torch.fx.GraphModule):
    """
    Removes all identity layers from the module.
    """

    class IdentityRemover(torch.fx.Transformer):
        def call_module(self, target, args, kwargs):
            if isinstance(self.submodules[target], nn.Identity):
                assert len(args) == 1
                return args[0]
            else:
                return super().call_module(target, args, kwargs)

    return IdentityRemover(gm).transform()


def fuse_conv_bn(gm: torch.fx.GraphModule, inplace=False):
    """
    Fuses Convolution/BN layers for inference purposes.
    """
    modules_patterns = [
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
    ]
    module_function_patterns = [
        (torch.nn.Conv1d, F.batch_norm),
        (torch.nn.Conv2d, F.batch_norm),
        (torch.nn.Conv3d, F.batch_norm),
    ]
    modules = dict(gm.named_modules())
    for pattern in modules_patterns:
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                eval_mode = all(not n.training for n in [conv, bn])
                if not eval_mode:
                    continue
                if not bn.track_running_stats:
                    continue
                fused_conv = fuse_conv_bn_eval(conv, bn)
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
    gm.graph.lint()
    for pattern in module_function_patterns:
        for node in gm.graph.nodes:
            if matches_module_function_pattern(pattern, node, modules):
                # TODO: support kwargs.
                if len(node.args) != 8:
                    continue
                conv = modules[node.args[0].target]
                bn_training = node.args[5]
                bn_eps = node.args[7]
                if conv.training or bn_training:
                    continue
                if type(bn_eps) is not float:
                    continue
                bn_args_is_constant = all(
                    n.op == "get_attr" and len(n.users) == 1 for n in node.args[1:5]
                )
                if not bn_args_is_constant:
                    continue
                bn_running_mean = fetch_attr(node.args[1].target, gm)
                bn_running_var = fetch_attr(node.args[2].target, gm)
                bn_weight = fetch_attr(node.args[3].target, gm)
                bn_bias = fetch_attr(node.args[4].target, gm)
                if bn_running_mean is None or bn_running_var is None:
                    continue
                fused_conv = copy.deepcopy(conv)
                fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
                    fused_conv.weight,
                    fused_conv.bias,
                    bn_running_mean,
                    bn_running_var,
                    bn_eps,
                    bn_weight,
                    bn_bias,
                )
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()

    return gm


def _philox_rand_like_meta(input, seed, offset):
    return _prims.TensorMeta(input)


def _philox_rand_like(input, seed, offset):
    # placeholder only used in tracing
    return torch.rand_like(input)


class NormalizedLinearNode:
    def __init__(self, node: torch.fx.Node) -> None:
        assert node.op == "call_function"
        assert node.target in [torch.nn.functional.linear]
        self.node: torch.fx.Node = node

    def get_input(self) -> torch.fx.Node:
        if len(self.node.args) > 0:
            return self.node.args[0]
        else:
            return self.node.kwargs["input"]

    def get_weight(self) -> torch.fx.Node:
        if len(self.node.args) > 1:
            return self.node.args[1]
        else:
            return self.node.kwargs["weight"]

    def get_bias(self) -> torch.fx.Node:
        if len(self.node.args) > 2:
            return self.node.args[2]
        else:
            return self.node.kwargs["bias"]


class NormalizedMatmulNode:
    def __init__(self, node: torch.fx.Node) -> None:
        assert node.op == "call_function"
        assert node.target in [torch.bmm, torch.matmul]
        self.node: torch.fx.Node = node

    def get_input(self) -> torch.fx.Node:
        if len(self.node.args) > 0:
            return self.node.args[0]
        else:
            return self.node.kwargs["input"]

    def get_other(self) -> torch.fx.Node:
        if len(self.node.args) > 1:
            return self.node.args[1]
        else:
            return self.node.kwargs["other"]


def check_permute(node: torch.fx.Node):
    ranks = len(node.meta["tensor_meta"].shape)
    if len(node.args) > 3:
        permutation = [node.args[i] % ranks for i in range(1, ranks + 1)]
    elif (
        "permutation" in node.kwargs
        and node.kwargs["permutation"] is not None
        and len(node.kwargs["permutation"]) > 2
    ):
        permutation = [i % ranks for i in node.kwargs["permutation"]]
    else:
        return False
    allowed_permutation = list(range(ranks))
    allowed_permutation[-1] = ranks - 2
    allowed_permutation[-2] = ranks - 1
    return permutation == allowed_permutation


def sink_cat_after_pointwise(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    def one_user(node):
        users = list(node.users)
        return users[0] if len(users) == 1 else None

    def is_view(node):
        view = {"view"}
        return node.op == "call_method" and node.target in view

    def is_pointwise_unary(node):
        pointwise = {torch.relu, torch.tanh, "relu", "tanh"}
        return node.op in {"call_function", "call_method"} and node.target in pointwise

    g = module.graph
    for node in g.nodes:
        if node.op != "call_function" or node.target != torch.cat:
            continue

        cat_or_view = node
        while True:
            user = one_user(cat_or_view)
            if not user or not is_view(user):
                break
            cat_or_view = user

        if user and is_pointwise_unary(user):
            with g.inserting_before(node):
                new_args = (
                    [
                        g.create_node(
                            user.op, user.target, args=(arg,), kwargs=user.kwargs
                        )
                        for arg in node.args[0]
                    ],
                )
                node.args = new_args
                user.replace_all_uses_with(cat_or_view)
                g.erase_node(user)
    g.lint()
    module.recompile()
    return module


def linear_permute_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.nodes:
        if (
            node.op == "call_method"
            and node.target == "permute"
            and check_permute(node)
        ):
            if len(node.args) > 0:
                input_node = node.args[0]
            else:
                input_node = node.kwargs["input"]
            if (
                input_node.op == "call_function"
                and input_node.target == torch.nn.functional.linear
            ):
                normalized = NormalizedLinearNode(input_node)
                input = normalized.get_input()
                weight = normalized.get_weight()
                bias = normalized.get_bias()
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(
                        linear_transpose, args=(input, weight, bias)
                    )
                    node.replace_all_uses_with(fused_node)
                    module.graph.erase_node(node)
                    if len(input_node.users) == 0:
                        module.graph.erase_node(input_node)

    module.graph.lint()
    module.recompile()
    return module


# Y1 = X * W^T + bias
# Y2 = Y1.permute(0, 2, 1)
# ---->
# Y2 = (W * X^T + bias.unsqueeze(-1))^T
def linear_transpose(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return torch.matmul(weight, input.transpose(-1, -2)) + bias.unsqueeze(-1)


def permute_linear_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.nodes:
        if node.op == "call_function" and node.target == torch.nn.functional.linear:
            if len(node.args) > 0:
                input_node = node.args[0]
            else:
                input_node = node.kwargs["input"]
            if (
                input_node.op == "call_method"
                and input_node.target == "permute"
                and check_permute(input_node)
            ):
                normalized = NormalizedLinearNode(node)
                if len(input_node.args) > 0:
                    input = input_node.args[0]
                else:
                    input = input_node.kwargs["input"]
                weight = normalized.get_weight()
                bias = normalized.get_bias()
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(
                        transpose_linear, args=(input, weight, bias)
                    )
                    node.replace_all_uses_with(fused_node)
                    module.graph.erase_node(node)
                    if len(input_node.users) == 0:
                        module.graph.erase_node(input_node)

    module.graph.lint()
    module.recompile()
    return module


def permute_matmul_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.nodes:
        if node.op == "call_function" and (
            node.target == torch.bmm or node.target == torch.matmul
        ):
            normalized = NormalizedMatmulNode(node)
            input_A_node = normalized.get_input()
            input_B_node = normalized.get_other()
            input_A = input_A_node
            input_B = input_B_node
            Atrans = Btrans = False
            if (
                input_A_node.op == "call_method"
                and input_A_node.target == "permute"
                and check_permute(input_A_node)
            ):
                Atrans = True
                if len(input_A_node.args) > 0:
                    input_A = input_A_node.args[0]
                else:
                    input_A = input_A_node.kwargs["input"]

            if (
                input_B_node.op == "call_method"
                and input_B_node.target == "permute"
                and check_permute(input_B_node)
            ):
                Btrans = True
                if len(input_B_node.args) > 0:
                    input_B = input_B_node.args[0]
                else:
                    input_B = input_B_node.kwargs["input"]

            if Atrans or Btrans:
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(
                        transpose_matmul,
                        args=(input_A, input_B, Atrans, Btrans),
                    )
                node.replace_all_uses_with(fused_node)
                module.graph.erase_node(node)
                if Atrans and len(input_A_node.users) == 0:
                    module.graph.erase_node(input_A_node)
                if Btrans and len(input_B_node.users) == 0:
                    module.graph.erase_node(input_B_node)

    module.graph.lint()
    module.recompile()
    return module


# X1 = X.permute(0, 2, 1)
# Y1 = X1 * W1^T + bias1
# ---->
# Y2 = X1.transpose(-1, -2) * W1^T + bias1
def transpose_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return torch.matmul(input.transpose(-1, -2), weight.t()) + bias


def transpose_matmul(A: torch.Tensor, B: torch.Tensor, Atrans: bool, Btrans: bool):
    if Atrans:
        A = A.transpose(-1, -2)
    if Btrans:
        B = B.transpose(-1, -2)
    return torch.matmul(A, B)


philox_rand_like = _prims._make_prim(
    schema="philox_rand_like(Tensor input, Tensor seed, int offset) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_philox_rand_like_meta,
    impl_aten=_philox_rand_like,
    doc="",
)


def _philox_seed_like_meta(x):
    return _prims.TensorMeta(_philox_seed_like(x))


def _philox_seed_like(x):
    # we need a tensor input here so AOT autograd properly captures this
    # with just a device input, this becomes a constant
    return torch.tensor(random.randrange(2**31), device=x.device, dtype=torch.int32)


philox_seed_like = _prims._make_prim(
    schema="philox_seed_like(Tensor other) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_philox_seed_like_meta,
    impl_aten=_philox_seed_like,
    doc="",
)


def null_ref():
    return None


class PhiloxRandomState:
    next_offset = 0
    seed = {}
    last_tracer_ref = null_ref

    @classmethod
    def reset(cls, tracer=None):
        cls.next_offset = 0
        cls.seed = {}
        cls.last_tracer_ref = weakref.ref(tracer) if tracer is not None else null_ref

    @classmethod
    def get_seed_offset(cls, x):
        modes = torch.fx.experimental.proxy_tensor.get_torch_dispatch_modes()
        proxy_modes = [m for m in modes if isinstance(m, ProxyTorchDispatchMode)]
        if proxy_modes:
            tracer = proxy_modes[0].tracer
            if cls.last_tracer_ref() is not tracer:
                # tracer changed, need to reset state
                cls.reset(tracer)
        else:
            # no tracer, need to reset state
            cls.reset()

        device = x.device
        if device not in cls.seed:
            # Compute the seed just once per trace so that we pass fewer
            # things from forward to backward
            cls.seed[device] = philox_seed_like(x)

        seed = cls.seed[device]
        offset = cls.next_offset
        cls.next_offset += x.numel()
        return seed, offset


class LowmemDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p
        scale = float(1.0 / (1.0 - p))
        seed, offset = PhiloxRandomState.get_seed_offset(x)
        ctx.save_for_backward(seed)
        ctx.offset = offset
        bool_mask = philox_rand_like(x, seed, offset) > p
        return bool_mask.to(x.dtype) * x * scale

    @staticmethod
    def backward(ctx, grad_output):
        p = ctx.p
        scale = float(1.0 / (1.0 - p))
        (seed,) = ctx.saved_tensors
        bool_mask = philox_rand_like(grad_output, seed, ctx.offset) > p
        return bool_mask.to(grad_output.dtype) * grad_output * scale, None


@torch.fx.wrap
def lowmem_dropout(input, p=0.5, training=True, inplace=False):
    if isinstance(input, torch.fx.Proxy):
        # double check we don't FX trace this
        return input.tracer.create_proxy(
            "call_function",
            lowmem_dropout,
            (input, p, training),
            {},
        )
    if not training or p == 0:
        return input
    result = LowmemDropout.apply(input, p)
    if inplace:
        input.copy_(result)
    return result


@torch.fx.wrap
def rand_like(x, **kwargs):
    if isinstance(x, torch.fx.Proxy):
        # double check we don't FX trace this
        return x.tracer.create_proxy("call_function", rand_like, (x), kwargs)
    assert kwargs.get("device", x.device) == x.device
    seed, offset = PhiloxRandomState.get_seed_offset(x)
    return philox_rand_like(x, seed, offset).to(kwargs.get("dtype", torch.float32))


replacements = {torch.nn.functional.dropout: lowmem_dropout, torch.rand_like: rand_like}
# Keep track of any replacement functions that use triton random,
# so they can be avoided when fallback_random is set
replacements_using_triton_random = {lowmem_dropout, rand_like}


'''
Quantization part for experiment
TODO: Maybe move to a separate file
'''
def is_quantized_graph_module(gm: torch.fx.GraphModule):
    found_quantize = False
    quantize_ops = (
        torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_channel,
    )
    for node in gm.graph.nodes:
        if node.target in quantize_ops:
            found_quantize = True
            break
    return found_quantize

def fuse_quantization(gm: torch.fx.GraphModule):
    # skip if gm is not a quantized graph module
    if not is_quantized_graph_module(gm):
        return gm

    print("model before fuse_reference_quantized_conv_relu is: {}".format(gm), flush=True)

    # gm = fuse_reference_quantized_conv_relu(gm)
    gm = fuse_reference_quantized_conv_relu_v2(gm)

    print("model after fuse_reference_quantized_conv_relu is: {}".format(gm), flush=True)

    return gm

def fuse_reference_quantized_conv_relu(gm: torch.fx.GraphModule):
    """
    For experiment
    Currently, quantized.convNd_prepck and quantized.convNd cannot be traced by meta tensor
    and cannot be lowered either.
    """
    aten = torch.ops.aten
    quantized_decomposed = torch.ops.quantized_decomposed
    convolution = aten.convolution.default
    relu = aten.relu.default
    quantize_per_tensor = quantized_decomposed.quantize_per_tensor
    dequantize_per_tensor = quantized_decomposed.dequantize_per_tensor
    quantize_per_channel = quantized_decomposed.quantize_per_channel
    dequantize_per_channel = quantized_decomposed.dequantize_per_channel

    def pattern(
        qx, x_scale, x_zp, x_quant_min, x_quant_max, x_dtype,
        qw, w_scale, w_zp, w_axis, w_quant_min, w_quant_max, w_dtype,
        bias, stride, padding, dilation, is_transposed, out_padding, groups,
            y_scale, y_zp, y_quant_min, y_quant_max, y_dtype):
        x = dequantize_per_tensor(qx, x_scale, x_zp, x_quant_min, x_quant_max, x_dtype)
        w = dequantize_per_channel(qw, w_scale, w_zp, w_axis, w_quant_min, w_quant_max, w_dtype)
        y = convolution(x, w, bias, stride, padding, dilation, is_transposed, out_padding, groups)
        y = relu(y)
        qy = quantize_per_tensor(y, y_scale, y_zp, y_quant_min, y_quant_max, y_dtype)
        return qy

    def replacement(
        qx, x_scale, x_zp, x_quant_min, x_quant_max, x_dtype,
        qw, w_scale, w_zp, w_axis, w_quant_min, w_quant_max, w_dtype,
        bias, stride, padding, dilation, is_transposed, out_padding, groups,
            y_scale, y_zp, y_quant_min, y_quant_max, y_dtype):
        # TODO: aten.convolution can be used for all conv1d/2d/3d but the spatial dim info
        # is lost. Here we hardcode conv2d for experiment.
        quantized = torch.ops.quantized
        w_packed = quantized.conv2d_prepack(qw, bias, stride, padding, dilation, groups)
        qy = quantized_decomposed.conv2d_relu(qx, w_packed, y_scale, y_zp, qw, stride, padding, dilation, groups)
        return qy

    subgraph_rewriter.replace_pattern(gm, pattern, replacement)
    gm.graph.lint()
    gm.recompile()

    print("graph after graph_rewriter is: {}".format(gm), flush=True)

    # TODO, actually do the prepack and add it as a call_attr node in the graph
    for n in gm.graph.nodes:
        print("node is graph is: {}".format(n), flush=True)
        print("node.target is: {}".format(n.target), flush=True)
        if n.target == torch.ops.quantized.conv2d_relu.new:
            print("---- hit torch.ops.quantized.conv2d_relu.new ----", flush=True)
        if n.target == torch.ops.quantized.conv2d_prepack:
            print("---- hit torch.ops.quantized.conv2d_prepack ----", flush=True)
            # res = torch.ops.quantized.conv2d_prepack()
            for args in n.args:
                print(args)
            # n.args[0] is quantize_per_channel

            print("n.args[0].args[0] is: {}".format(n.args[0].args[0]))
            print(n.args[0].args[0].op)
            print(n.args[0].args[0].target)
            # print(getattr(gm, n.args[0].args[0].target))
            print(type(getattr(gm, n.args[0].args[0].target)))
            
            print(getattr(gm, n.args[0].args[0].target))
            W_q = torch.quantize_per_channel(
                getattr(gm, n.args[0].args[0].target), torch.tensor([1.0], dtype=torch.float, device=torch.device("cpu")),
                torch.tensor([0], dtype=torch.float, device=torch.device("cpu")).long(), 0,
                dtype=torch.qint8)
            # res = torch.ops.quantized.conv2d_prepack(n.args[0], n.args[1], n.args[2], n.args[3], n.args[4], n.args[5])


            print("n.args[2] is: {}".format(n.args[2]), flush=True)

            res = torch.ops.quantized.conv2d_prepack(W_q, None, n.args[2], n.args[3], n.args[4], n.args[5])
            setattr(gm, 'prepacked_weight', res) # set attr to graph module
            print("res is: {}".format(res), flush=True)
            print("res is: {}".format(type(res)), flush=True)

            origin_w, origin_b = torch.ops.quantized.conv2d_unpack(res)

            print("origin_w is: {}".format(origin_w), flush=True)
            print("origin_b is: {}".format(origin_b), flush=True)

            # example_inputs = torch.randn(1, 1, 224, 224)
            # qx = torch.quantize_per_tensor(example_inputs, 1.0, 0, dtype=torch.quint8)
            # qy = torch.ops.quantized.conv2d_relu.new(qx, res, 1.0, 0)
            # print("qy is: {}".format(qy))

            
            assert list(n.users).__len__() == 1, "prepack node only has one user node as conv"
            conv_node = list(n.users)[0]
            print("conv_node.target is: {}".format(conv_node.target), flush=True)

            with gm.graph.inserting_before(conv_node):
                # packed_weight = gm.graph.create_node("get_attr", res, tuple(prepack_args), {})
                packed_weight = gm.graph.get_attr("prepacked_weight", )
            n.replace_all_uses_with(packed_weight)
            gm.graph.erase_node(n)

    gm.graph.lint()
    gm.recompile()

    # # Replace torch.ops.quantized.conv2d_relu.new to torch.ops.quantized_decomposed.conv2d_relu
    # for n in gm.graph.nodes:
    #     print("node is graph is: {}".format(n), flush=True)
    #     print("node.target is: {}".format(n.target), flush=True)
    #     if n.target == torch.ops.quantized.conv2d_relu.new:
    #         n.target = torch.ops.quantized_decomposed.conv2d_relu

    # gm.graph.lint()
    # gm.recompile()    

    return gm


def fuse_reference_quantized_conv_relu_v2(gm: torch.fx.GraphModule):
    """
    For experiment
    Currently, quantized.convNd_prepck and quantized.convNd cannot be traced by meta tensor
    and cannot be lowered either.
    """
    aten = torch.ops.aten
    quantized_decomposed = torch.ops.quantized_decomposed
    convolution = aten.convolution.default
    relu = aten.relu.default
    quantize_per_tensor = quantized_decomposed.quantize_per_tensor
    dequantize_per_tensor = quantized_decomposed.dequantize_per_tensor
    quantize_per_channel = quantized_decomposed.quantize_per_channel
    dequantize_per_channel = quantized_decomposed.dequantize_per_channel

    def pattern(
        qx, x_scale, x_zp, x_quant_min, x_quant_max, x_dtype,
        qw, w_scale, w_zp, w_axis, w_quant_min, w_quant_max, w_dtype,
        bias, stride, padding, dilation, is_transposed, out_padding, groups,
            y_scale, y_zp, y_quant_min, y_quant_max, y_dtype):
        x = dequantize_per_tensor(qx, x_scale, x_zp, x_quant_min, x_quant_max, x_dtype)
        w = dequantize_per_channel(qw, w_scale, w_zp, w_axis, w_quant_min, w_quant_max, w_dtype)
        y = convolution(x, w, bias, stride, padding, dilation, is_transposed, out_padding, groups)
        y = relu(y)
        qy = quantize_per_tensor(y, y_scale, y_zp, y_quant_min, y_quant_max, y_dtype)
        return qy

    def replacement(
        qx, x_scale, x_zp, x_quant_min, x_quant_max, x_dtype,
        qw, w_scale, w_zp, w_axis, w_quant_min, w_quant_max, w_dtype,
        bias, stride, padding, dilation, is_transposed, out_padding, groups,
            y_scale, y_zp, y_quant_min, y_quant_max, y_dtype):
        # TODO: aten.convolution can be used for all conv1d/2d/3d but the spatial dim info
        # is lost. Here we hardcode conv2d for experiment.
        # quantized = torch.ops.quantized
        # w_packed = quantized.conv2d_prepack(qw, bias, stride, padding, dilation, groups)
        
        # qy = quantized_decomposed.conv2d_relu_v2(qx, qw, bias, stride, padding, dilation, groups, y_scale, y_zp)

        qy = quantized_decomposed.conv2d_relu_v2(qx, qw, bias, stride, padding, dilation, groups, y_scale, y_zp,
        x_scale, x_zp, w_scale, w_zp, w_axis)

        return qy

    subgraph_rewriter.replace_pattern(gm, pattern, replacement)
    gm.graph.lint()
    gm.recompile()

    return gm
