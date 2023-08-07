import copy
import functools
import itertools
import operator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.utils import (
    get_bias_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_weight_qspec,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
    get_source_partitions,
    SourcePartition,
)
from .quantizer import (
    OperatorConfig,
    OperatorPatternType,
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from .xnnpack_quantizer import _is_annotated

__all__ = [
    "X86InductorQuantizer",
    "get_default_x86_inductor_quantization_config",
]


@dataclass
class _X86InductorQuantizationAnnotation(QuantizationAnnotation):
    # _is_output_of_quantized_pattern:
    #  * Node as output node of a fusion pattern.
    #  * The fusion pattern supports int8 data type.
    #  * The fusion pattern has inputs annotated to insert observer.
    _is_output_of_quantized_pattern: bool = False


# Ops support int8 data type and excludes ops like conv, linear.
quantizable_ops_pt2e: Set = {
    torch.ops.aten.max_pool2d_with_indices.default,
    torch.ops.aten.cat.default,
    torch.ops.aten.avg_pool2d.default,
}


# Ops that:
# 1. Ops prefer to run with int8 when int8 input is given.
# 2. Ops don't support int8 in and fp32 out.
int8_in_int8_out_ops_pt2e: Set = {
    torch.ops.aten.max_pool2d_with_indices.default,
    torch.ops.aten.cat.default,
    torch.ops.aten.avg_pool2d.default
}


def _is_node_annotated(_node):
    """
    return True if the node is annotated, otherwise return False
    """
    return (
        "quantization_annotation" in _node.meta
        and _node.meta["quantization_annotation"]._annotated
    )


def _is_any_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False.
    """
    return any(_is_node_annotated(node) for node in nodes)


def _is_all_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    return True if all of the node is annotated, otherwise return False.
    """
    return all(_is_node_annotated(node) for node in nodes)


def _is_quantized_op_pt2e(node: torch.fx.Node):
    """
    Used for pt2e flow to check if the node is a quantized node:
    Case1: the node has been annotated as output node of a fusion pattern.
    Case2: the node has been annotated as single quantized node.
    """
    if not _is_any_annotated([node]):
        # The node has not been annotated, directly return False
        return False
    quantization_annotation = node.meta.get("quantization_annotation", None)
    assert isinstance(quantization_annotation, _X86InductorQuantizationAnnotation)
    return quantization_annotation._is_output_of_quantized_pattern


def _supported_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    # TODO: Add more supported operators here.
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        "conv2d": [
            [torch.nn.Conv2d],
            [F.conv2d],
        ],
    }

    # Append Conv Optional(Add) Optioinal(ReLU)
    conv_add_relu_options = itertools.product(
        [torch.nn.Conv2d, F.conv2d],
        [torch.add, operator.add, None],  # add
        [torch.nn.ReLU, F.relu, None],  # relu
    )
    for conv_op, add_op, relu_op in conv_add_relu_options:
        if add_op is None:
            # Append Conv ReLU
            supported_operators["conv2d"].append([conv_op, relu_op])  # type: ignore[list-item]
        elif relu_op is None:
            # Append Conv Add
            supported_operators["conv2d"].append([conv_op, add_op])  # type: ignore[list-item]
        else:
            # Append Conv Add ReLU
            supported_operators["conv2d"].append([conv_op, add_op, relu_op])  # type: ignore[list-item]

    return copy.deepcopy(supported_operators)


def _get_supported_x86_inductor_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []
    for quantization_config in [
        get_default_x86_inductor_quantization_config(),
    ]:
        ops = _supported_quantized_operators()
        for pattern_list in ops.values():
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list)
            )
    return copy.deepcopy(supported_config_and_operators)


@functools.lru_cache
def get_default_x86_inductor_quantization_config():
    act_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        HistogramObserver
    )

    # Copy from x86 default qconfig from torch/ao/quantization/qconfig.py
    act_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,  # reduce_range=False
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            eps=2**-12
        ),
    )

    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        PerChannelMinMaxObserver
    )
    extra_args: Dict[str, Any] = {"eps": 2**-12}
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,  # 0 corresponding to weight shape = (oc, ic, kh, kw) of conv
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )
    bias_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        PlaceholderObserver
    )
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.float, observer_or_fake_quant_ctr=bias_observer_or_fake_quant_ctr
    )
    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
    )
    return quantization_config


def _get_supported_config_and_operators() -> List[OperatorConfig]:
    return _get_supported_x86_inductor_config_and_operators()


class X86InductorQuantizer(Quantizer):
    supported_config_and_operators = _get_supported_config_and_operators()

    def __init__(self):
        super().__init__()
        self.global_config: QuantizationConfig = None  # type: ignore[assignment]
        self.operator_type_config: Dict[str, Optional[QuantizationConfig]] = {}

    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        op_configs: Set[QuantizationConfig] = set({})
        for spec, _ in cls.supported_config_and_operators:
            op_configs.add(spec)
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(
        cls, quantization_config: Optional[QuantizationConfig]
    ) -> List[OperatorPatternType]:
        if quantization_config is None:
            all_ops = []
            for _, ops in cls.supported_config_and_operators:
                all_ops.extend(ops)
            return all_ops

        for config, ops in cls.supported_config_and_operators:
            if config == quantization_config:
                return ops
        return []

    def set_global(self, quantization_config: QuantizationConfig):
        self.global_config = quantization_config
        return self

    def set_config_for_operator_type(
        self, operator_type: str, quantization_config: QuantizationConfig
    ):
        self.operator_type_config[operator_type] = quantization_config
        return self

    def _annotate_conv_node_helper(
        self,
        conv_node: torch.fx.Node,
        annotate_output: bool,
        quantization_config: QuantizationConfig,
    ) -> None:
        """Helper function to annotate the conv node"""
        input_qspec_map = {}
        input_node = conv_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        weight_node = conv_node.args[1]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        bias_node = conv_node.args[2]
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        if annotate_output:
            conv_node.meta[
                "quantization_annotation"
            ] = _X86InductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                # TODO<leslie> Remove the annotate of output when oneDNN qconv support fp32 out.
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
        else:
            conv_node.meta[
                "quantization_annotation"
            ] = _X86InductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )

    def _get_output_nodes_of_partitions(
        self,
        partition_list: List[SourcePartition],
    ) -> List[torch.fx.Node]:
        """Helper function to get the output node list from partition list"""
        output_node_list = []
        for partition in partition_list:
            if len(partition.output_nodes) > 1:
                raise ValueError("Input partition has more than one output node")
            output_node = partition.output_nodes[0]
            assert isinstance(output_node, Node)
            output_node_list.append(output_node)
        if len(output_node_list) != len(partition_list):
            raise ValueError(
                "length of output_node_list should equal to length of partition_list"
            )
        return output_node_list

    def _get_input_idx_for_binary_node(
        self,
        conv_gemm_node: torch.fx.Node,
        binary_node: torch.fx.Node,
    ):
        """Helper function to check conv_gemm and extra input node index
        for binary node fused with conv_gemm.
        """
        conv_gemm_node_idx = None
        extra_input_node_idx = None
        if (binary_node.args[0].op == "call_function") and (  # type: ignore[union-attr]
            binary_node.args[0] == conv_gemm_node
        ):
            conv_gemm_node_idx = 0
            extra_input_node_idx = 1
        elif (binary_node.args[1].op == "call_function") and (  # type: ignore[union-attr]
            binary_node.args[1] == conv_gemm_node
        ):
            conv_gemm_node_idx = 1
            extra_input_node_idx = 0
        extra_input_node = binary_node.args[extra_input_node_idx]  # type: ignore[index]
        assert isinstance(extra_input_node, Node)
        return conv_gemm_node_idx, extra_input_node_idx

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        model = self._annotate_for_static_quantization_config(model)
        return model

    def _annotate_for_static_quantization_config(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        r"""
        High-level description of quantization recipe for X86 Inductor Backend:
        Step 1: Apply quantization recipe for fusion patterns of conv/linear to enable int8 data type actively.
        Step 2: Propagate quantization annotation for patterns besides conv/linear. Go through the pattern in model
        from start to the end. If a pattern supports computation with int8 data type and inputs connected to
        quantized patterns, annotate its inputs as quantized pattern.
        Step 3: Since in step 2, we only annotate the inputs of quantized pattern. For some quantized patterns,
        such as maxpool2d, which only supports output with int8 data type when the input is with int8 data type,
        we need to annotate the output of this pattern.
        """

        config = self.global_config

        # Step1: Recipe of fusion patterns like conv/linear.
        self._annotate_conv2d_fusion_pattern(model, config)

        # Step2: Recipe to propagate annotation for patterns beside conv/linear.
        # Go through all the nodes from start to end.
        # Recipe refer to https://github.com/intel/intel-extension-for-pytorch/blob/
        # 90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_recipe.py#L538
        for node in model.graph.nodes:
            self._annotation_propagation_quantizable_pattern(node, config)

        # Step3: For quantizable ops, such as maxpool2d, we need to quantize its output if it is quantized
        # in inputs. So, we can fuse dq-operator-q into a quantized op.
        # Refer to https://github.com/intel/intel-extension-for-pytorch/blob/
        # 90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_recipe.py#L487
        for node in model.graph.nodes:
            self._annotate_output_for_int8_in_int8_out_pattern(node, config)

        return model

    def _annotate_conv2d_fusion_pattern(
        self, model: torch.fx.GraphModule, config: QuantizationConfig
    ):
        self._annotate_conv2d_binary_unary(model, config)
        self._annotate_conv2d_binary(model, config)
        self._annotate_conv2d_unary(model, config)
        self._annotate_conv2d(model, config)

    def _annotate_conv2d_binary_unary(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        # Conv2d + add + unary op
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, operator.add, torch.nn.ReLU]
        )
        for fused_partition in fused_partitions:
            conv_partition, binary_partition, unary_partition = fused_partition
            conv_node, binary_node, unary_node = self._get_output_nodes_of_partitions(
                [conv_partition, binary_partition, unary_partition]
            )
            conv_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(
                conv_node, binary_node
            )
            if (conv_node_idx is None) or (extra_input_node_idx is None):
                continue
            if conv_node != binary_node.args[conv_node_idx]:
                raise ValueError(f"{conv_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.convolution.default
            ):
                # No conv node found to be fused with add
                continue
            if _is_annotated([unary_node, binary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            binary_node.meta[
                "quantization_annotation"
            ] = _X86InductorQuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map,
                _annotated=True,
            )
            unary_node.meta[
                "quantization_annotation"
            ] = _X86InductorQuantizationAnnotation(
                # TODO<leslie> Remove the annotate of output when oneDNN qconv support fp32 out.
                output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_conv2d_binary(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        # Conv2d + add
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, operator.add]
        )
        for fused_partition in fused_partitions:
            conv_partition, binary_partition = fused_partition
            conv_node, binary_node = self._get_output_nodes_of_partitions(
                [conv_partition, binary_partition]
            )
            conv_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(
                conv_node, binary_node
            )
            if (conv_node_idx is None) or (extra_input_node_idx is None):
                continue
            if conv_node != binary_node.args[conv_node_idx]:
                raise ValueError(f"{conv_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]
            assert isinstance(conv_node, Node)
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.convolution.default
            ):
                # No conv node found to be fused with add
                continue
            if _is_annotated([binary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            binary_node.meta[
                "quantization_annotation"
            ] = _X86InductorQuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map,
                # TODO<leslie> Remove the annotate of output when oneDNN qconv support fp32 out.
                output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_conv2d_unary(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, torch.nn.ReLU]
        )
        for fused_partition in fused_partitions:
            conv_partition, unary_partition = fused_partition
            conv_node, unary_node = self._get_output_nodes_of_partitions(
                [conv_partition, unary_partition]
            )
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.convolution.default
            ):
                continue
            if _is_annotated([unary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            unary_node.meta[
                "quantization_annotation"
            ] = _X86InductorQuantizationAnnotation(
                # TODO<leslie> Remove the annotate of output when oneDNN qconv support fp32 out.
                output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_conv2d(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        conv_partitions = get_source_partitions(
            gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d]
        )
        conv_partitions = list(itertools.chain(*conv_partitions.values()))
        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError("conv partition has more than one output node")
            conv_node = conv_partition.output_nodes[0]
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.convolution.default
            ):
                raise ValueError(f"{conv_node} is not an aten conv2d operator")
            # skip annotation if it is already annotated
            if _is_annotated([conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, True, quantization_config)

    def _annotate_maxpool2d(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        if node.target is not torch.ops.aten.max_pool2d_with_indices.default or not (
            len(list(node.users)) == 1
            and (list(node.users)[0].target == operator.getitem)
        ):
            return
        maxpool_node = node
        getitem_node = list(node.users)[0]
        if _is_any_annotated([getitem_node, maxpool_node]):
            return
        input_node = maxpool_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map = {}
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        maxpool_node.meta[
            "quantization_annotation"
        ] = _X86InductorQuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        getitem_node.meta[
            "quantization_annotation"
        ] = _X86InductorQuantizationAnnotation(
            _annotated=True,
            _is_output_of_quantized_pattern=True,
        )

    def _annotate_cat(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        cat_node = node
        input_nodes = cat_node.args[0]
        first_input_node = input_nodes[0]
        input_qspec_map = {}
        input_qspec_map[first_input_node] = get_input_act_qspec(quantization_config)
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec((first_input_node, cat_node))

        for input_node in input_nodes[1:]:
            input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

        cat_node.meta[
            "quantization_annotation"
        ] = _X86InductorQuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
            _is_output_of_quantized_pattern=True,
        )

    # def _annotate_avg_pool2d(
    #     self, node: Node, quantization_config: QuantizationConfig
    # ) -> None:
    #     avg_pool2d_node = node
    #     input_node = avg_pool2d_node.args[0]
    #     input_qspec_map = {}
    #     input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
    #     avg_pool2d_node.meta[
    #         "quantization_annotation"
    #     ] = _X86InductorQuantizationAnnotation(
    #         input_qspec_map=input_qspec_map,
    #         _annotated=True,
    #         _is_output_of_quantized_pattern=True,
    #     )

    def _annotation_propagation_quantizable_pattern(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        # Propagate annotation to quantizable patterns.
        if (
            (node.target in quantizable_ops_pt2e)
            and (not _is_any_annotated([node]))
            and (node.op == "call_function")
        ):

            def is_all_inputs_connected_to_quantized_op(input_nodes):
                # Ensure all the inputs connect to fusion pattern or quantized node
                for input_node in input_nodes:
                    if not _is_quantized_op_pt2e(input_node):
                        return False
                return True

            if node.target is torch.ops.aten.max_pool2d_with_indices.default:
                # Recipe of maxpool2d: check input arg[0] of maxpool2d is quantized or not
                input_nodes_to_check = [node.all_input_nodes[0]]
                if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                    return
                self._annotate_maxpool2d(node, quantization_config)
                return
            elif node.target is torch.ops.aten.cat.default:
                input_nodes_to_check = node.all_input_nodes
                if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                    return
                self._annotate_cat(node, quantization_config)
            else:
                input_node = node.all_input_nodes[0]
                if not is_all_inputs_connected_to_quantized_op([input_node,]):
                    return
                input_qspec_map = {}
                input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
                node.meta[
                    "quantization_annotation"
                ] = _X86InductorQuantizationAnnotation(
                    input_qspec_map=input_qspec_map,
                    _annotated=True,
                    _is_output_of_quantized_pattern=True,
                )
        return

    def _annotate_output_for_int8_in_int8_out_pattern(
        self, node: Node, quantization_config: QuantizationConfig
    ) -> None:
        r"""
        Check and insert observer at output of node in int8_in_int8_out_ops_pt2e if needed.
        Recipe refers to https://github.com/intel/intel-extension-for-pytorch/blob/
        90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_utils.py#L495
        """
        if (node.target in int8_in_int8_out_ops_pt2e) and (_is_any_annotated([node])):
            if node.target == torch.ops.aten.max_pool2d_with_indices.default:
                maxpool_node = node
                assert len(list(maxpool_node.users)) == 1 and (
                    list(maxpool_node.users)[0].target == operator.getitem
                )
                getitem_node = list(node.users)[0]
                if not _is_all_annotated([getitem_node, maxpool_node]):
                    return
                # Get the quantization_annotation from getitem_node
                getitem_quantization_annotation = (
                    getitem_node.meta["quantization_annotation"]
                    if "quantization_annotation" in getitem_node.meta
                    else None
                )
                if (
                    getitem_quantization_annotation
                    and getitem_quantization_annotation._is_output_of_quantized_pattern
                ):
                    # Annotate the output_qspec of getitem_node
                    input_act = maxpool_node.args[0]
                    assert isinstance(input_act, Node)
                    assert isinstance(maxpool_node, Node)
                    edge_or_node: Tuple[Node, Node] = (input_act, maxpool_node)
                    getitem_node.meta[
                        "quantization_annotation"
                    ].output_qspec = SharedQuantizationSpec(edge_or_node)
            elif node.target is torch.ops.aten.cat.default:
                cat_node = node
                cat_quantization_annotation = (
                    cat_node.meta["quantization_annotation"]
                    if "quantization_annotation" in cat_node.meta
                    else None
                )
                if (
                    cat_quantization_annotation
                    and cat_quantization_annotation._is_output_of_quantized_pattern
                ):
                    # Annotate the output_qspec of cat_node
                    first_input_node = cat_node.all_input_nodes[0]
                    assert isinstance(first_input_node, Node)
                    assert isinstance(cat_node, Node)
                    edge_or_node: Tuple[Node, Node] = (first_input_node, cat_node)
                    cat_node.meta[
                        "quantization_annotation"
                    ].output_qspec = SharedQuantizationSpec(edge_or_node)
            elif node.target is torch.ops.aten.avg_pool2d.default:
                avg_pool2d_node = node
                avg_pool2d_quantization_annotation = (
                    avg_pool2d_node.meta["quantization_annotation"]
                    if "quantization_annotation" in avg_pool2d_node.meta
                    else None
                )
                if (
                    avg_pool2d_quantization_annotation
                    and avg_pool2d_quantization_annotation._is_output_of_quantized_pattern
                ):
                    # Annotate the output_qspec of cat_node
                    input_node = avg_pool2d_node.all_input_nodes[0]
                    assert isinstance(input_node, Node)
                    assert isinstance(avg_pool2d_node, Node)
                    edge_or_node: Tuple[Node, Node] = (input_node, avg_pool2d_node)
                    avg_pool2d_node.meta[
                        "quantization_annotation"
                    ].output_qspec = SharedQuantizationSpec(edge_or_node)
            else:
                # TODO <leslie>: Enable recipes for more int8_in_int8_out_ops
                pass
        return

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators
