class Evaluator:

    def __init__(self):
        self.time_list = []
        self.avg_time = 0

    def evaluate(self, evaluate_data_dict):
        all_outputs = evaluate_data_dict[0]
        all_inputs = evaluate_data_dict[1]
        time = evaluate_data_dict[2]
        self.time_list.append(time)

        self._evaluate(all_outputs, all_inputs)

    def summary(self):
        self.avg_time = sum(self.time_list) / len(self.time_list)
        self._summary()

    def _evaluate(self, all_outputs, all_inputs):
        raise NotImplementedError

    def _summary(self):
        raise NotImplementedError
