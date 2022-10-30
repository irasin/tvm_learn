#%%
"""
modify src onnx model

1. rewrite batchsize
2. infer shape

Attention:
1. all inputs/outputs batchszie dim will be modified together, which means some NLP/Audio models will introduce problems maybe(#FIXME)


"""
import os

import onnx
import onnx.helper as helper
from onnx import shape_inference, numpy_helper

## for reference
# ONNX_DTYPE = {
#     0: onnx.TensorProto.FLOAT,
#     1: onnx.TensorProto.FLOAT,
#     2: onnx.TensorProto.UINT8,
#     3: onnx.TensorProto.INT8,
#     4: onnx.TensorProto.UINT16,
#     5: onnx.TensorProto.INT16,
#     6: onnx.TensorProto.INT32,
#     7: onnx.TensorProto.INT64,
#     8: onnx.TensorProto.STRING,
#     9: onnx.TensorProto.BOOL,
# }


def modify_tensor_dim(tensor, dim_value_dict):
    tensor_shape = tensor.type.tensor_type.shape.dim

    new_shape = []
    for i in range(len(tensor_shape)):
        if tensor_shape[i].dim_value:
            new_shape.append(tensor_shape[i].dim_value)
        else:
            new_shape.append(None)

    for dim, value in dim_value_dict.items():
        assert dim < len(new_shape)
        new_shape[dim] = value

    assert None not in new_shape

    new_tensor = helper.make_tensor_value_info(
        name=tensor.name,
        elem_type=tensor.type.tensor_type.elem_type,
        shape=new_shape)

    return new_tensor


def modify_batch_size(model, batch_size, modify_reshape_dim):

    ## rewrite input and output
    graph = model.graph
    initializer = graph.initializer
    inputs = graph.input
    outputs = graph.output

    dim_value_dict = {0: batch_size}

    new_inputs = [modify_tensor_dim(i, dim_value_dict) for i in inputs]
    new_outputs = [modify_tensor_dim(i, dim_value_dict) for i in outputs]

    while inputs:
        inputs.pop()
    inputs.extend(new_inputs)

    while outputs:
        outputs.pop()
    outputs.extend(new_outputs)

    ## we may need to modify reshape initializer if we modify input batchsize
    ## but this may introduce some other problems when the purpose of reshape operations are totally different(#FIXME)
    if modify_reshape_dim:
        for idx, i in enumerate(initializer):
            if "Reshape" in i.name:
                shape = numpy_helper.to_array(i).copy()
                shape[0] = batch_size
                initializer[idx].CopyFrom(
                    numpy_helper.from_array(shape, i.name))

    ## infer shape
    ## sometimes we need add initializer as input
    # input_size = len(inputs)
    # for i, tensor in enumerate(initializer):
    #     value_info = helper.make_tensor_value_info(tensor.name, tensor.type.tensor_type.elem_type, tensor.dims)
    #     inputs.insert(i + input_size, value_info)

    inferred_model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred_model)

    ## remove initializer
    # while len(inferred_model.graph.input) != input_size:
    #     inferred_model.graph.input.pop()

    return inferred_model


def modify_onnx(src_onnx_path,
                dst_onnx_path=None,
                batch_size=1,
                modify_reshape_dim=True):

    if dst_onnx_path is None:
        dst_onnx_path = f"{os.path.splitext(src_onnx_path)}_batchsize_{batch_size}.onnx"

    model = onnx.load(src_onnx_path)
    model = modify_batch_size(model, batch_size, modify_reshape_dim)

    ## extend this function to add other modification

    onnx.save(model, dst_onnx_path)

    return model


if __name__ == "__main__":

    modify_onnx(
        src_onnx_path=
        "/home/chen.chen/igie/tests/test_iluvatar/end2end/yolov5m/yolov5m.onnx",
        dst_onnx_path=
        "/home/chen.chen/igie/tests/test_iluvatar/end2end/yolov5m/tmp.onnx",
        batch_size=40,
    )

# %%
