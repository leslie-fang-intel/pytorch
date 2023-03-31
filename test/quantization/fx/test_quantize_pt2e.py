# Owner(s): ["oncall: quantization"]
import torch
import torch.nn as nn
import torch._dynamo as torchdynamo
from torch.testing._internal.common_utils import xfailIfPython311
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
    skipIfNoONEDNN,
    skipIfNoX86,
)
from torch.testing._internal.common_quantization import NodeSpec as ns
from torch.testing._internal.common_quantized import (
    override_quantized_engine,
)
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
    observer,
)
from torch.ao.quantization.backend_config import (
    get_qnnpack_backend_config,
)
from torch.ao.quantization.backend_config._qnnpack_pt2e import get_qnnpack_pt2e_backend_config
from torch.ao.quantization.backend_config._x86_inductor_pt2e import get_x86_inductor_pt2e_backend_config
from torch.ao.quantization.backend_config.x86 import get_x86_backend_config
from torch.ao.quantization.quantize_fx import prepare_fx, convert_to_reference_fx, convert_fx
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.ns.fx.utils import (
    compute_sqnr,
)
import copy
import itertools
from torch._inductor.compile_fx import compile_fx


@skipIfNoQNNPACK
class TestQuantizePT2E(QuantizationTestCase):
    @xfailIfPython311
    def test_qconfig_none(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        with override_quantized_engine("qnnpack"):
            m = M().eval()
            example_inputs = (torch.randn(1, 1, 1, 1),)
            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            qconfig = get_default_qconfig("qnnpack")
            qconfig_mapping = QConfigMapping().set_global(qconfig) \
                                              .set_module_name("conv2", None)
            backend_config = get_qnnpack_pt2e_backend_config()
            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
            m(*example_inputs)
            m = convert_pt2e(m)
            m(*example_inputs)

            # first conv is quantized, second conv is not quantized
            node_occurrence = {
                # two for input of the first conv, one for output for the first conv
                ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
            }
            node_list = [
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.aten.convolution.default),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.aten.convolution.default),
            ]
            self.checkGraphModuleNodes(
                m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    @xfailIfPython311
    def test_qconfig_module_type(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.linear = nn.Linear(9, 3)

            def forward(self, x):
                x = self.conv(x)
                x = x.reshape((1, -1))
                x = self.linear(x)
                return x

        with override_quantized_engine("qnnpack"):
            m = M().eval()
            example_inputs = (torch.randn(1, 1, 3, 3),)

            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            qconfig = get_default_qconfig("qnnpack")
            qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Conv2d, qconfig)
            backend_config = get_qnnpack_pt2e_backend_config()
            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
            m(*example_inputs)
            m = convert_pt2e(m)
            m(*example_inputs)
            # conv is quantized, linear is not quantized
            node_occurrence = {
                # two for input and weight of the conv, one for output for the conv
                ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
            }
            node_list = [
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.aten.convolution.default),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.aten.addmm.default),
            ]
            self.checkGraphModuleNodes(m, expected_node_list=node_list)

    @xfailIfPython311
    def test_rearrange_weight_observer_for_decomposed_linear(self):
        """
        Check whether weight observer is correctly rearranged for decomposed linear.
        before:
            weight - t - observer \
              input - observer - addmm/mm
        after:
            weight - observer - t \
              input - observer - addmm/mm
        """
        class M(torch.nn.Module):
            def __init__(self, with_bias, use_relu):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=with_bias)
                self.relu = nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                x = self.linear(x)
                return self.relu(x) if self.use_relu else x

        with_bias_list = [True, False]
        use_relu_list = [True, False]
        cases = itertools.product(with_bias_list, use_relu_list)
        for with_bias, use_relu in cases:
            m = M(with_bias, use_relu).eval()
            example_inputs = (torch.randn(1, 4),)

            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            qconfig = get_default_qconfig('qnnpack')
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            backend_config = get_qnnpack_pt2e_backend_config()
            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)

            # 1. Check graph nodes:
            # - args[0] of t should be the weight observer
            # - args[-1] of addmm/mm should be t
            error_msg = 'Weight observer is not correctly rearranged for decomposed linear'
            for node in m.graph.nodes:
                if node.target == torch.ops.aten.t.default:
                    target = node.args[0].target
                    self.assertTrue(isinstance(getattr(m, target), observer.ObserverBase), error_msg)
                elif node.target in (torch.ops.aten.addmm.default, torch.ops.aten.mm.default):
                    target = node.args[-1].target
                    self.assertTrue(target == torch.ops.aten.t.default, error_msg)

            # 2. Check m.code to ensure `m.recompile()` is called.
            # If weight observer is rearranged in graph but `m.recompile()` is not called,
            # m.code would be wrong.
            code_before_recompile = m.code
            m.recompile()
            code_after_recompile = m.code
            self.assertTrue(code_before_recompile == code_after_recompile, error_msg)

@skipIfNoQNNPACK
class TestQuantizePT2EX86Inductor(QuantizationTestCase):
    @skipIfNoX86
    @skipIfNoONEDNN
    @xfailIfPython311
    def test_inductor_backend_config_conv(self):
        class M(torch.nn.Module):
            def __init__(self, use_relu: bool = False, inplace_relu: bool = False):
                super().__init__()
                self.use_relu = use_relu
                self.conv1 = nn.Conv2d(3, 6, (2, 2), stride=(1, 1), padding=(1, 1))
                self.relu = nn.ReLU(inplace=inplace_relu)

            def forward(self, x):
                x = self.conv1(x)
                return self.relu(x) if self.use_relu else x

        use_relu_list = [True, False]
        inplace_relu_list = [True, False]
        with override_quantized_engine("x86"):
            with torch.no_grad():
                for use_relu, inplace_relu in itertools.product(use_relu_list, inplace_relu_list):
                    m = M(use_relu=use_relu, inplace_relu=inplace_relu).eval()
                    example_inputs = (torch.randn(2, 3, 4, 4),)
                    # program capture
                    # **TODO** Add testcase for tracing_mode="symbolic" after fix issue:
                    # https://github.com/pytorch/pytorch/issues/96274
                    export_module, guards = torchdynamo.export(
                        m,
                        *copy.deepcopy(example_inputs),
                        aten_graph=True,
                        tracing_mode="real",
                    )

                    qconfig = get_default_qconfig("x86")
                    qconfig_mapping = QConfigMapping().set_global(qconfig)
                    backend_config = get_x86_inductor_pt2e_backend_config()
                    prepare_module = prepare_pt2e(export_module, qconfig_mapping, example_inputs, backend_config)
                    prepare_module(*example_inputs)
                    convert_module = convert_pt2e(prepare_module)
                    convert_module(*example_inputs)

                    # Fake quant should only be inserted at start and end
                    node_occurrence = {
                        # one for input and weight of the conv, one for output for the conv
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 2,
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 1,
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 1,
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 2,
                    }
                    if use_relu:
                        node_list = [
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                            ns.call_function(torch.ops.aten.convolution.default),
                            ns.call_function(torch.ops.aten.relu_.default if inplace_relu else torch.ops.aten.relu.default),
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                        ]
                    else:
                        node_list = [
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                            ns.call_function(torch.ops.aten.convolution.default),
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                        ]
                    self.checkGraphModuleNodes(convert_module,
                                               expected_node_occurrence=node_occurrence,
                                               expected_node_list=node_list)

                    # Step1: Ref result in 1.X fx path
                    backend_config_1_x = get_x86_backend_config()
                    m_copy = copy.deepcopy(m)
                    m_prepare_fx = prepare_fx(m_copy, qconfig_mapping, example_inputs, backend_config=backend_config_1_x)
                    after_prepare_result_fx = m_prepare_fx(*example_inputs)
                    m_convert_fx = convert_fx(m_prepare_fx, backend_config=backend_config_1_x)
                    ref_result = m_convert_fx(*example_inputs)

                    # Step2: Start to lowering into Inductor
                    run = compile_fx(convert_module, example_inputs)
                    # Inductor first run
                    inductor_res = run(*example_inputs)
                    # Inductor second run
                    inductor_res = run(*example_inputs)
                    self.assertEqual(ref_result, inductor_res, atol=5e-2, rtol=5e-2)

    @skipIfNoX86
    @skipIfNoONEDNN
    @xfailIfPython311
    def test_inductor_backend_config_conv_add_relu(self):
        class Mod(torch.nn.Module):
            def __init__(self, inplace_add=False, inplace_relu=False) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
                self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
                self.relu = torch.nn.ReLU(inplace=inplace_relu)
                self.inplace_add = inplace_add
                self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
                self.conv5 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

            def forward(self, x):
                if not self.inplace_add:
                    x1 = self.conv(x)
                    relu_res = self.relu(self.conv2(x1) + self.conv3(x1))
                    res = self.conv5(self.conv4(relu_res)) + relu_res
                    return res
                else:
                    x1 = self.conv(x)
                    accum = self.conv2(x1)
                    accum += self.conv3(x1)
                    relu_res = self.relu(accum)
                    relu_res += self.conv5(self.conv4(relu_res))
                    return relu_res

        inplace_add_list = [True, False]
        inplace_relu_list = [True, False]
        with override_quantized_engine("x86"):
            with torch.no_grad():
                for inplace_add, inplace_relu in itertools.product(inplace_add_list, inplace_relu_list):
                    m = Mod(inplace_add=inplace_add, inplace_relu=inplace_relu).eval()
                    example_inputs = (torch.randn(2, 3, 16, 16),)
                    # program capture
                    # **TODO** Add testcase for tracing_mode="symbolic" after fix issue:
                    # https://github.com/pytorch/pytorch/issues/96274
                    export_module, guards = torchdynamo.export(
                        m,
                        *copy.deepcopy(example_inputs),
                        aten_graph=True,
                        tracing_mode="real",
                    )

                    qconfig = get_default_qconfig("x86")
                    qconfig_mapping = QConfigMapping().set_global(qconfig)
                    backend_config = get_x86_inductor_pt2e_backend_config()
                    prepare_module = prepare_pt2e(export_module, qconfig_mapping, example_inputs, backend_config)
                    prepare_module(*example_inputs)
                    convert_module = convert_pt2e(prepare_module)
                    convert_module(*example_inputs)
                    # Step2: Start to lowering into Inductor
                    run = compile_fx(convert_module, example_inputs)
                    # Inductor first run
                    inductor_res = run(*example_inputs)
                    # Inductor second run
                    inductor_res = run(*example_inputs)

                    # Fake quant should only be inserted at start and end
                    node_occurrence = {
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 6,
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 5,
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 5,
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 6,
                    }
                    self.checkGraphModuleNodes(convert_module,
                                               expected_node_occurrence=node_occurrence)

    @skipIfNoX86
    @skipIfNoONEDNN
    @xfailIfPython311
    def test_inductor_conv_with_weight_unpack(self):
        torch._inductor.config.cpp.enable_kernel_profile = True
        import copy
        from torch import _dynamo, _inductor
        from torch._inductor import config
        import logging
        import numpy as np
        import random

        local_seed = 2023
        torch.manual_seed(local_seed) # Set PyTorch seed
        np.random.seed(seed=local_seed) # Set Numpy seed
        random.seed(local_seed) # Set the Python seed

        torch._dynamo.config.log_level = logging.DEBUG
        torch._dynamo.config.verbose = True
        torch._inductor.config.trace.enabled = True
        torch._inductor.config.debug = True

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    # in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
                    in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                return self.relu(x)

        torch.backends.quantized.engine = "x86"
        # example_inputs = (torch.randn(1, 1, 224, 224),)
        example_inputs = (torch.randn(1, 3, 16, 16).to(memory_format=torch.channels_last),)
        m = Mod().eval()
        # program capture
        
        
        tracing_mode = "real"
        #tracing_mode = "symbolic"
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
            tracing_mode=tracing_mode,
        )

        m = m.eval()
        print("model after torchdynamo export is: {}".format(m), flush=True)
        print("guards is: {}".format(guards), flush=True)

        # m(torch.randn(2, 3, 16, 16).to(memory_format=torch.channels_last))
        # exit(-1)

        backend_config = get_x86_inductor_pt2e_backend_config()
        qconfig = get_default_qconfig("x86")
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        before_fusion_result = m(*example_inputs)

        m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
        after_prepare_result = m(*example_inputs)
        print("model after prepare_pt2e is: {}".format(m), flush=True)

        print("check the result after prepare: {}".format(
            torch.allclose(before_fusion_result, after_prepare_result)),
            flush=True)
        
        m = convert_pt2e(m)
        after_quant_result = m(*example_inputs)
        print("model after convert_pt2e is: {}".format(m), flush=True)

        print("check the result after convert: {}".format(
            torch.allclose(before_fusion_result, after_quant_result, rtol=5e-02, atol=5e-02)),
            flush=True)

        run = compile_fx(m, example_inputs)

        print("start the first run", flush=True)
        inductor_result = run(*example_inputs)

        print("start the second run", flush=True)
        inductor_result = run(*example_inputs)

        assert torch.allclose(before_fusion_result, inductor_result, rtol=5e-02, atol=5e-02)

class TestQuantizePT2EModels(QuantizationTestCase):
    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    @xfailIfPython311
    def test_resnet18(self):
        import torchvision
        with override_quantized_engine("qnnpack"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = torchvision.models.resnet18().eval()
            m_copy = copy.deepcopy(m)
            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            backend_config = get_qnnpack_pt2e_backend_config()
            # TODO: define qconfig_mapping specifically for executorch
            qconfig = get_default_qconfig("qnnpack")
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            before_fusion_result = m(*example_inputs)

            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)

            # checking that we inserted observers correctly for maxpool operator (input and
            # output share observer instance)
            self.assertEqual(id(m.activation_post_process_3), id(m.activation_post_process_2))
            after_prepare_result = m(*example_inputs)
            m = convert_pt2e(m)

            after_quant_result = m(*example_inputs)

            # comparing with existing fx graph mode quantization reference flow
            backend_config = get_qnnpack_backend_config()
            m_fx = prepare_fx(m_copy, qconfig_mapping, example_inputs, backend_config=backend_config)
            after_prepare_result_fx = m_fx(*example_inputs)
            m_fx = convert_to_reference_fx(m_fx, backend_config=backend_config)

            after_quant_result_fx = m_fx(*example_inputs)

            # the result matches exactly after prepare
            self.assertEqual(after_prepare_result, after_prepare_result_fx)
            self.assertEqual(compute_sqnr(after_prepare_result, after_prepare_result_fx), torch.tensor(float("inf")))
            # there are slight differences after convert due to different implementations
            # of quant/dequant
            self.assertTrue(torch.max(after_quant_result - after_quant_result_fx) < 1e-1)
            self.assertTrue(compute_sqnr(after_quant_result, after_quant_result_fx) > 35)
