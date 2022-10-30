import time
from contextlib import contextmanager

import tvm


class Timer:

    def __init__(self):
        self.last_duration = 0  # ms
        self.duration_list = []  # ms

    def _update(self, duration):
        self.last_duration = duration
        self.duration_list.append(self.last_duration)

    # @contextmanager
    # def timeit_sync(self, device):
    #     try:
    #         get_timer = tvm.get_global_func("profiling.get_timer")
    #         start = tvm.get_global_func("profiling.start")
    #         stop = tvm.get_global_func("profiling.stop")
    #         elapse_time = tvm.get_global_func("profiling.elapse_time")

    #         timer = get_timer(device)
    #         device.sync()
    #         start(timer)

    #         yield

    #         stop(timer)
    #         self._update(elapse_time(timer) / 1e6)  ## ns / 1e6 -> ms

    #     except:
    #         pass

    @contextmanager
    def timeit(self):
        t1 = time.time()

        yield

        t2 = time.time()
        self._update((t2 - t1) * 1e3)  ## s * 1e3 -> ms
