from torch._dynamo import register_backend
from torch.ao.quantization._pt2e.quantizer import  X86InductorQuantizer
import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
import torch
import torch._dynamo as torchdynamo
import copy
from torch.fx.experimental.proxy_tensor import make_fx
from torch import _guards
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.exc import CondOpArgsMismatchError, ResetRequired, UserError, UserErrorType
from torch.ao.quantization._quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e_quantizer,
    prepare_qat_pt2e_quantizer,
)

@register_backend
def inductor(*args, **kwargs):
    # do import here to avoid loading inductor into memory when it is not used
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(*args, **kwargs)

fail_graph_count = 0
total_graph_count = 0
@register_backend
def inductor_dynamic_quant(*args, **kwargs):
    # do import here to avoid loading inductor into memory when it is not used
    from torch._inductor.compile_fx import compile_fx
    print("---- inside inductor_dynamic_quant backend ----", flush=True)
    print(len(args), flush=True)
    # print("kwargs: {}".format(kwargs.empty()))
    print(args, flush=True)
    print(kwargs, flush=True)

    model = args[0]
    example_inputs = args[1]

    print("model is: {}".format(model), flush=True)
    
    global total_graph_count
    global fail_graph_count

    total_graph_count += 1
    print("total_graph_count is: {}".format(total_graph_count), flush=True)

    with torch.no_grad():
        # program capture
        # export_model, guards = torchdynamo.export(
        #     model,
        #     *copy.deepcopy(example_inputs),
        #     aten_graph=True,
        # )
        try:
            decomposition_table = None
            # Step1: generate the Aten graph
            fake_mode = _guards.detect_fake_mode(example_inputs)
            example_fake_inputs = [fake_mode.from_tensor(t) for t in example_inputs]
            graph = model
            def graph_with_interpreter(*args):
                with torch.fx.traceback.preserve_node_meta():
                    return torch.fx.Interpreter(graph).run(*args)
            with enable_python_dispatcher(), fake_mode:
                # try:
                graph = make_fx(
                    graph_with_interpreter,
                    decomposition_table=decomposition_table,
                    tracing_mode="real",
                    _allow_non_fake_inputs=True,
                    pre_autograd=False,
                )(*example_fake_inputs)
                # except CondOpArgsMismatchError as e:
                #     # Wrap the internal error to the user-facing error
                #     # raise UserError(UserErrorType.DYNAMIC_CONTROL_FLOW, str(e))
                #     fail_graph_count += 1
                #     print("fail_graph_count is: {}".format(fail_graph_count), flush=True)
                #     return model
            model = graph
            quantizer = X86InductorQuantizer()
            operator_config = xiq.get_default_x86_inductor_quantization_config(is_dynamic=True)
            # operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
            quantizer.set_global(operator_config)
            # Step2: Dynamic quant prepare
            prepared_model = prepare_pt2e_quantizer(model, quantizer)
            # Step3: Dynamic quant Convert
            quantized_model = convert_pt2e(prepared_model)
            args = list(args)
            args[0] = quantized_model
            args = tuple(args)
            return compile_fx(*args, **kwargs)
        except:
            fail_graph_count += 1
            print("fail_graph_count is: {}".format(fail_graph_count), flush=True)
            return compile_fx(*args, **kwargs)
