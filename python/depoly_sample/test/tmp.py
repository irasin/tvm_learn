import time

import sys

# timer = tvm.get_global_func("profiling.get_timer")(device)
sys.path.append("..")

from fast_deploy.misc.target import get_target
from fast_deploy.misc.timer import Timer

_, device = get_target("llvm")

timer = Timer()
# with timer.timeit_sync(device):
#     time.sleep(0.4)
# print(timer.duration_list)

with timer.timeit():
    time.sleep(0.7)
print(timer.duration_list)

# _, device = get_target("cuda")

# with timer.timeit_sync(device):
#     time.sleep(0.4)
# print(timer.duration_list)
