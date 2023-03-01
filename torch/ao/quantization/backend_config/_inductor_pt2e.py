'''
Based on _qnnpack_pt2e.py
'''

import operator
import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    DTypeConfig,
    ObservationType,
    BackendPatternConfig,
)

weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)
from typing import List
from torch.ao.quantization.utils import MatchAllNode
import itertools

def get_linear_configs():
    linear_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]

    # TODO: need to fix the way we insert observers for this pattern
    # should be solved in the new fusion API
    # reason that this doesn't work: the pattern is a bit complicated and we don't
    # have a way to specify which input of the pattern we would like to observe
    # pattern:
    # bias input weight
    # \     |    /
    #  \    |   t
    #   \   |  /
    #    addmm
    # we want to observe "weight" as weight, but there is not way to convey this
    # information with current pattern language
    #
    # right now:
    # original:
    #         weight - t \
    #         input  - addmm
    # observed (no hack):
    #      weight - t - observer \
    #       input - observer - addmm
    # target:
    #      weight - observer - t \
    #        input - observer - addmm

    # def root_node_getter(node_pattern):
    #     addmm, bias, act, weight = node_pattern
    #     return addmm

    # linear_configs.append(
    #     BackendPatternConfig((torch.ops.aten.addmm.default, MatchAllNode, MatchAllNode, torch.ops.aten.t.default))
    #     .set_observation_type(observation_type)  # noqa: E131
    #     .set_dtype_configs(dtype_configs)
    #     ._set_root_node_getter(root_node_getter))

    linear_configs.append(
        BackendPatternConfig(torch.ops.aten.addmm.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 0})
    )
    linear_configs.append(
        BackendPatternConfig((torch.ops.aten.addmm.default, torch.ops.aten.relu.default))
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 0})
    )
    linear_configs.append(
        BackendPatternConfig(torch.ops.aten.mm.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1})
    )
    linear_configs.append(
        BackendPatternConfig((torch.ops.aten.mm.default, torch.ops.aten.relu.default))
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1})
    )
    return linear_configs

def get_conv_configs():
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    conv_configs.append(
        BackendPatternConfig(torch.ops.aten.convolution.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    conv_configs.append(
        BackendPatternConfig((torch.ops.aten.convolution.default, torch.ops.aten.relu.default))
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    # TODO: remove when functionalization is supported in PT2 mode
    conv_configs.append(
        BackendPatternConfig((torch.ops.aten.convolution.default, torch.ops.aten.relu_.default))
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    # Conv add ReLU case
    def _conv_add_relu_root_node_getter_left(pattern):
        relu, add_pattern = pattern
        _, conv, _ = add_pattern
        return conv
    def _conv_add_relu_extra_inputs_getter_left(pattern):
        """ get inputs pattern for extra inputs, inputs for root node
        are assumed to be copied over from root node to the fused node
        """
        relu, add_pattern = pattern
        _, conv, extra_input = add_pattern
        return [extra_input]
    
    conv_add_relu_optioins = itertools.product(
        [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor],  # add op
        [torch.ops.aten.relu.default, torch.ops.aten.relu_.default],  # relu op
    )
    for add_op, relu_op in conv_add_relu_optioins:
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((relu_op, (add_op, torch.ops.aten.convolution.default, MatchAllNode)))  # noqa: E131
                .set_observation_type(observation_type)
                .set_dtype_configs(dtype_configs)
                ._set_input_type_to_index({"weight": 1, "bias": 2})
                ._set_root_node_getter(_conv_add_relu_root_node_getter_left)
                ._set_extra_inputs_getter(_conv_add_relu_extra_inputs_getter_left)
                # ._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping)
        )
    def _conv_add_relu_root_node_getter_right(pattern):
        relu, add_pattern = pattern
        _, _, conv = add_pattern
        return conv
    def _conv_add_relu_extra_inputs_getter_right(pattern):
        """ get inputs pattern for extra inputs, inputs for root node
        are assumed to be copied over from root node to the fused node
        """
        relu, add_pattern = pattern
        _, extra_input, conv = add_pattern
        return [extra_input]
    
    conv_add_relu_optioins = itertools.product(
        [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor],  # add op
        [torch.ops.aten.relu.default, torch.ops.aten.relu_.default],  # relu op
    )
    for add_op, relu_op in conv_add_relu_optioins:
        conv_configs.append(
            BackendPatternConfig()
                ._set_pattern_complex_format((relu_op, (add_op, MatchAllNode, torch.ops.aten.convolution.default)))  # noqa: E131
                .set_observation_type(observation_type)
                .set_dtype_configs(dtype_configs)
                ._set_input_type_to_index({"weight": 1, "bias": 2})
                ._set_root_node_getter(_conv_add_relu_root_node_getter_right)
                ._set_extra_inputs_getter(_conv_add_relu_extra_inputs_getter_right)
                # ._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping)
        )
    # # Conv add case
    # def _conv_add_root_node_getter_left(pattern):
    #     _, conv, _ = pattern
    #     return conv
    # def _conv_add_extra_inputs_getter_left(pattern):
    #     """ get inputs pattern for extra inputs, inputs for root node
    #     are assumed to be copied over from root node to the fused node
    #     """
    #     _, conv, extra_input = pattern
    #     return [extra_input]
    # for add_op in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
    #     conv_configs.append(
    #         BackendPatternConfig()
    #             ._set_pattern_complex_format((add_op, torch.ops.aten.convolution.default, MatchAllNode))  # noqa: E131
    #             .set_observation_type(observation_type)
    #             .set_dtype_configs(dtype_configs)
    #             ._set_input_type_to_index({"weight": 1, "bias": 2})
    #             ._set_root_node_getter(_conv_add_root_node_getter_left)
    #             ._set_extra_inputs_getter(_conv_add_extra_inputs_getter_left)
    #             # ._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping)
    #     )
    return conv_configs

def get_pooling_configs():
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]

    def root_node_getter(node_pattern):
        getitem, maxpool, index = node_pattern
        return maxpool

    backend_pattern_configs.append(
        BackendPatternConfig()
        ._set_pattern_complex_format((operator.getitem, torch.ops.aten.max_pool2d_with_indices.default, 0))
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_root_node_getter(root_node_getter)
    )

    return backend_pattern_configs

def get_relu_configs():
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    backend_pattern_configs.append(
        BackendPatternConfig(torch.ops.aten.relu.default)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs))
    return backend_pattern_configs

def get_binary_op_configs():
    binary_op_configs: List[BackendPatternConfig] = []
    dtype_configs = [weighted_op_quint8_dtype_config]
    num_tensor_args_to_observation_type_mapping = {
        # TODO: this is not used right now since we have extra check in prepare
        # will need to change this to NO_OBSERVER later after we implemented
        # Tensor dtype inference properly
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    for op_with_quantized_bop_scalar_variant in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
        bop_patterns = [
            (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu.default),
            op_with_quantized_bop_scalar_variant,
            # TODO: remove when functionalization is supported in pt2_mode
            (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu_.default),
        ]
        for bop_pattern in bop_patterns:
            binary_op_configs.append(
                BackendPatternConfig(bop_pattern)
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    ._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping))

    return binary_op_configs

def get_inductor_pt2e_backend_config():
    return (
        BackendConfig("inductor_pytorch_2.0_export")
        .set_backend_pattern_configs(get_conv_configs())
        .set_backend_pattern_configs(get_linear_configs())
        .set_backend_pattern_configs(get_binary_op_configs())
        .set_backend_pattern_configs(get_pooling_configs())
        .set_backend_pattern_configs(get_relu_configs())
    )
