import tvm


def get_target(target_name):
    if target_name == "iluvatar":
        target = tvm.target.iluvatar(model="MR")
    elif target_name == "iluvatar_cudnn":
        target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas")
    elif target_name == "llvm":
        target = tvm.target.Target(target_name)
    elif target_name == "cuda":
        target = tvm.target.Target(target_name)
    else:
        raise Exception(f"Unsupport Target name: {target_name}!")

    device = tvm.device(target.kind.name, 0)

    return target, device
