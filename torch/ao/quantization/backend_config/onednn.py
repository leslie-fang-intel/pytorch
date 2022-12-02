import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.functional as F
import torch.nn.quantized._reference as nnqr
from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_bn_configs,
    _get_cat_config,
    _get_conv_configs,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_linear_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
)
from .backend_config import (
    BackendPatternConfig,
    BackendConfig,
    DTypeConfig,
    ObservationType,
)
from ..fuser_method_mappings import (
    reverse_sequential_wrapper2,
    reverse3,
)
import operator
import torch.nn.functional as F

from torch.ao.quantization.utils import MatchAllNode
# ===================
# |  DTYPE CONFIGS  |
# ===================

onednn_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

onednn_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

onednn_default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

onednn_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

conv_dtype_configs = [onednn_weighted_op_int8_dtype_config]
linear_dtype_configs = [
    onednn_weighted_op_int8_dtype_config,
    onednn_default_dynamic_int8_dtype_config,
]
binary_op_dtype_configs = [onednn_weighted_op_int8_dtype_config]
default_op_dtype_configs = [onednn_default_op_quint8_dtype_config]
fixed_qparams_op_dtype_configs = [onednn_weighted_op_int8_dtype_config]
share_qparams_op_dtype_configs = [onednn_default_op_quint8_dtype_config]
rnn_op_dtype_configs = [
    onednn_default_dynamic_int8_dtype_config,
]
embedding_op_dtype_configs = [
    onednn_weight_only_quint8_dtype_config,
]

# ===================
# |  FUSER METHODS  |
# ===================

def _fuse_linear_bn_leaky_relu(is_qat, linear, bn, leaky_relu):
    r"""Given the linear, bn and leaky_relu modules, fuses them and returns the fused module
    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
                or post training quantization fusion
        linear: Module instance of type Linear
        bn: BatchNorm1d instance that needs to be fused with the linear layer
        leaky_relu: LeakyReLU instance that needs to be fused with the linear layer
    Examples::
        >>> m1 = nn.Linear(20, 10)
        >>> b1 = nn.BatchNorm1d(10)
        >>> lr = nn.LeakyReLU(0.01)
        >>> m2 = _fuse_linear_bn_leaky_relu(m1, b1, lr)
    """
    assert(linear.training == bn.training and bn.training == leaky_relu.training),\
        "Linear, BN and LeakyReLU all must be in the same mode (train or eval)."

    if is_qat:
        raise NotImplementedError("Cannot fuse train modules: {}".format((linear, bn, leaky_relu)))
    else:
        map_to_fused_module_eval = {
            nn.Linear: nni.LinearLeakyReLU,
        }
        fused_module = map_to_fused_module_eval.get(type(linear), None)
        if fused_module is not None:
            fused_linear = nn.utils.fusion.fuse_linear_bn_eval(linear, bn)
            fm = fused_module(fused_linear, leaky_relu)
            return fm
        else:
            raise NotImplementedError("Cannot fuse eval modules: {}".format((linear, bn, leaky_relu)))

# ======================
# |  CONFIGS FOR CONV  |
# ======================

observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT

conv_configs = _get_conv_configs(conv_dtype_configs)

# **Important** We need to define pattern: torch.add, nn.Conv2d, MatchAllNode before pattern: torch.add, MatchAllNode, nn.Conv2d
# For the case of pattern: torch.add, nn.Conv2d, nn.Conv2d
# 1. The first defined pattern will be matched firstly.
# 2. In the get_fuser_method_new of pytorch/torch/ao/quantization/fuser_method_mappings.py,
#    pattern: torch.add, nn.Conv2d, nn.Conv2d will matche to pattern: torch.add, MatchAllNode, nn.Conv2d
#    or pattern: torch.add, nn.Conv2d, MatchAllNode. And pattern: torch.add, nn.Conv2d, MatchAllNode has been 
#    tested firstly (Due to the implementation of itertools.product).
#    So to avoid choose of wrong fused method, we need these order to defined pattern.

##----------------- For Conv Add fusion, we only need below code ----------------------##
# conv   Y
#   \   /
#    add

# include:
# conv   conv
#   \   /
#    add

def fuse_conv_add_right(is_qat, add, conv, _):
    # add, _, conv = add_pattern
    return nni.Conv2dAdd(add, conv)
    #return conv

def conv_add_root_node_getter_right(pattern):
    _, conv, _ = pattern
    return conv

def conv_add_extra_inputs_getter_right(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _, conv, extra_input = pattern
    return [extra_input]

conv_configs.append(
    BackendPatternConfig((torch.add, nn.Conv2d, MatchAllNode))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_add_right)
        ._set_root_node_getter(conv_add_root_node_getter_right)
        ._set_extra_inputs_getter(conv_add_extra_inputs_getter_right)
        .set_fused_module(nni.Conv2dAdd))

conv_configs.append(
    BackendPatternConfig((operator.add, nn.Conv2d, MatchAllNode))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_add_right)
        ._set_root_node_getter(conv_add_root_node_getter_right)
        ._set_extra_inputs_getter(conv_add_extra_inputs_getter_right)
        .set_fused_module(nni.Conv2dAdd))

# conv + bn + add fusion
# conv
#   \
#    bn     Y
#      \   /
#       add
def fuse_conv_bn_add_right(is_qat, add, bn_conv, _):
    # add, _, conv = add_pattern
    bn, conv = bn_conv
    fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
    return nni.Conv2dAdd(add, fused_conv)
    #return conv

def conv_bn_add_root_node_getter_right(pattern):
    _, bn_conv, _ = pattern
    bn, conv = bn_conv
    return conv

def conv_bn_add_extra_inputs_getter_right(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _, bn_conv, extra_input = pattern
    bn, conv = bn_conv
    return [extra_input]

conv_configs.append(
    BackendPatternConfig((torch.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_bn_add_right)
        ._set_root_node_getter(conv_bn_add_root_node_getter_right)
        ._set_extra_inputs_getter(conv_bn_add_extra_inputs_getter_right)
        .set_fused_module(nni.Conv2dAdd))

conv_configs.append(
    BackendPatternConfig((operator.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_bn_add_right)
        ._set_root_node_getter(conv_bn_add_root_node_getter_right)
        ._set_extra_inputs_getter(conv_bn_add_extra_inputs_getter_right)
        .set_fused_module(nni.Conv2dAdd))

#  Y    conv
#   \   /
#    add
def fuse_conv_add(is_qat, add, _, conv):
    # add, _, conv = add_pattern
    return nni.Conv2dAdd(add, conv)
    #return conv

def conv_add_root_node_getter(pattern):
    add, _, conv = pattern
    return conv

def conv_add_extra_inputs_getter(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _, extra_input, conv = pattern
    return [extra_input]

conv_configs.append(
    BackendPatternConfig((torch.add, MatchAllNode, nn.Conv2d))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_add)
        ._set_root_node_getter(conv_add_root_node_getter)
        ._set_extra_inputs_getter(conv_add_extra_inputs_getter)
        .set_fused_module(nni.Conv2dAdd))

conv_configs.append(
    BackendPatternConfig((operator.add, MatchAllNode, nn.Conv2d))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_add)
        ._set_root_node_getter(conv_add_root_node_getter)
        ._set_extra_inputs_getter(conv_add_extra_inputs_getter)
        .set_fused_module(nni.Conv2dAdd))

conv_configs.append(
    BackendPatternConfig(nni.Conv2dAdd)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(conv_dtype_configs)
        .set_root_module(nn.Conv2d)
        .set_reference_quantized_module(nnqr.Conv2d))


##----------------- Split Line: Below for conv add relu fusion ------------------##

##--------------- For RN50 conv_add_relu fusion, we only needs below codes ---------------##
# conv   Y
#   \   /
#    add
#     \
#     relu
def fuse_conv_add_relu_right(is_qat, relu, add_pattern):
    add, conv, _ = add_pattern
    return nni.Conv2dAddRelu(relu, add, conv)
    #return conv

def conv_add_relu_root_node_getter_right(pattern):
    relu, add_pattern = pattern
    _, conv, _ = add_pattern
    return conv

def conv_add_relu_extra_inputs_getter_right(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    relu, add_pattern = pattern
    _, conv, extra_input = add_pattern
    return [extra_input]

conv_configs.append(
    BackendPatternConfig((nn.ReLU, (torch.add, nn.Conv2d, MatchAllNode)))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_add_relu_right)
        ._set_root_node_getter(conv_add_relu_root_node_getter_right)
        ._set_extra_inputs_getter(conv_add_relu_extra_inputs_getter_right)
        .set_fused_module(nni.Conv2dAddRelu))

conv_configs.append(
    BackendPatternConfig((nn.ReLU, (operator.add, nn.Conv2d, MatchAllNode)))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_add_relu_right)
        ._set_root_node_getter(conv_add_relu_root_node_getter_right)
        ._set_extra_inputs_getter(conv_add_relu_extra_inputs_getter_right)
        .set_fused_module(nni.Conv2dAddRelu))

# Conv + BN + Add + Relu
# conv
#  \
#  bn   Y
#   \   /
#    add
#     \
#     relu
def fuse_conv_bn_add_relu_right(is_qat, relu, add_pattern):
    add, bn_conv, _ = add_pattern
    bn, conv = bn_conv
    fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
    return nni.Conv2dAddRelu(relu, add, fused_conv)
    #return conv

def conv_bn_add_relu_root_node_getter_right(pattern):
    relu, add_pattern = pattern
    _, bn_conv, _ = add_pattern
    bn, conv = bn_conv
    return conv

def conv_bn_add_relu_extra_inputs_getter_right(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    relu, add_pattern = pattern
    _, bn_conv, extra_input = add_pattern
    bn, conv = bn_conv
    return [extra_input]

conv_configs.append(
    BackendPatternConfig((nn.ReLU, (torch.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode)))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_bn_add_relu_right)
        ._set_root_node_getter(conv_bn_add_relu_root_node_getter_right)
        ._set_extra_inputs_getter(conv_bn_add_relu_extra_inputs_getter_right)
        .set_fused_module(nni.Conv2dAddRelu))

conv_configs.append(
    BackendPatternConfig((nn.ReLU, (operator.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode)))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_bn_add_relu_right)
        ._set_root_node_getter(conv_bn_add_relu_root_node_getter_right)
        ._set_extra_inputs_getter(conv_bn_add_relu_extra_inputs_getter_right)
        .set_fused_module(nni.Conv2dAddRelu))

#  Y   conv
#   \   /
#    add
#     \
#     relu
def fuse_conv_add_relu(is_qat, relu, add_pattern):
    add, _, conv = add_pattern
    return nni.Conv2dAddRelu(relu, add, conv)
    #return conv

def conv_add_relu_root_node_getter(pattern):
    relu, add_pattern = pattern
    _, _, conv = add_pattern
    return conv

def conv_add_relu_extra_inputs_getter(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    relu, add_pattern = pattern
    _, extra_input, conv = add_pattern
    return [extra_input]

conv_configs.append(
    BackendPatternConfig((nn.ReLU, (torch.add, MatchAllNode, nn.Conv2d)))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_add_relu)
        ._set_root_node_getter(conv_add_relu_root_node_getter)
        ._set_extra_inputs_getter(conv_add_relu_extra_inputs_getter)
        .set_fused_module(nni.Conv2dAddRelu))

conv_configs.append(
    BackendPatternConfig((nn.ReLU, (operator.add, MatchAllNode, nn.Conv2d)))
        .set_observation_type(observation_type)
        .set_dtype_configs(conv_dtype_configs)
        .set_fuser_method(fuse_conv_add_relu)
        ._set_root_node_getter(conv_add_relu_root_node_getter)
        ._set_extra_inputs_getter(conv_add_relu_extra_inputs_getter)
        .set_fused_module(nni.Conv2dAddRelu))

conv_configs.append(
    BackendPatternConfig(nni.Conv2dAddRelu)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(conv_dtype_configs)
        .set_root_module(nn.Conv2d)
        .set_reference_quantized_module(nnqr.Conv2d))
# ========================
# |  CONFIGS FOR LINEAR  |
# ========================

linear_configs = _get_linear_configs(linear_dtype_configs)

# (1) Linear + leaky_relu
# -------------------
# 1.1 linear module + leaky_relu fusion config
# linear leaky_relu, linear module + leaky_relu module
linear_configs.append(
    BackendPatternConfig((nn.LeakyReLU, nn.Linear))
        .set_dtype_configs(linear_dtype_configs)  # noqa: E131
        .set_fuser_method(reverse_sequential_wrapper2(nni.LinearLeakyReLU))
        .set_fused_module(nni.LinearLeakyReLU))
# linear leaky_relu, linear module + functional leaky_relu
linear_configs.append(
    BackendPatternConfig((F.leaky_relu, nn.Linear))
        .set_dtype_configs(linear_dtype_configs)  # noqa: E131
        .set_fuser_method(reverse_sequential_wrapper2(nni.LinearLeakyReLU))
        .set_fused_module(nni.LinearLeakyReLU))
# linear leaky_relu, linear module + BN + leaky_relu
linear_configs.append(
    BackendPatternConfig((nn.LeakyReLU, (nn.BatchNorm1d, nn.Linear)))
        .set_dtype_configs(linear_dtype_configs)  # noqa: E131
        .set_fuser_method(reverse3(_fuse_linear_bn_leaky_relu))
        .set_fused_module(nni.LinearLeakyReLU))

# 1.2 linear module + leaky_relu, fused module configs
# linear leaky_relu, fused module
linear_configs.append(
    BackendPatternConfig(nni.LinearLeakyReLU)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(linear_dtype_configs)
        .set_root_module(nn.Linear)
        .set_reference_quantized_module(nnqr.Linear))
# 1.3 functional linear + leaky_relu configs
# linear leaky_relu, functional linear + leaky_relu module
linear_configs.append(
    BackendPatternConfig((nn.LeakyReLU, F.linear))
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(linear_dtype_configs))
# linear leaky_relu, functional linear + functional leaky_relu
linear_configs.append(
    BackendPatternConfig((F.leaky_relu, F.linear))
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(linear_dtype_configs))

# =====================
# |  BACKEND CONFIGS  |
# =====================

def get_onednn_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native ONEDNN backend.
    """
    return BackendConfig("onednn") \
        .set_backend_pattern_configs(conv_configs) \
        .set_backend_pattern_configs(linear_configs) \
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))

__all__ = [
    "get_onednn_backend_config",
]
