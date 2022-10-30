import os
import onnx
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
from .onnx_modifer import modify_onnx

from fast_deploy.misc.target import get_target


def build_lib(onnx_model, input_shape_dict, target):
    mod, params = relay.frontend.from_onnx(onnx_model,
                                           input_shape_dict,
                                           freeze_params=True)

    ##TODO: since we have not intergrated the mod optimization into relay.build
    # we may need to add some pass here manually, for example
    # mod = relay.transform.DynamicToStatic()(mod)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    return lib


def onnx2igie(onnx_model_path, input_shape_dict, batch_size=1, target="llvm"):
    path_with_suffix = os.path.splitext(onnx_model_path)[0]
    path_with_suffix = f"{path_with_suffix}_batch_size_{batch_size}_{target}"

    modified_model_path = path_with_suffix + ".onnx"
    igie_lib_path = path_with_suffix + ".so"
    print("load model from ", modified_model_path)
    target, device = get_target(target)
    if not os.path.exists(igie_lib_path):

        if not os.path.exists(modified_model_path):

            onnx_model = modify_onnx(onnx_model_path, modified_model_path,
                                     batch_size, True)
        else:
            onnx_model = onnx.load(modified_model_path)

        lib = build_lib(onnx_model, input_shape_dict, target)
        lib.export_library(igie_lib_path)

    else:
        print("load from ", igie_lib_path)
        lib = tvm.runtime.load_module(igie_lib_path)

    m = graph_executor.GraphModule(lib["default"](device))
    return m, device
