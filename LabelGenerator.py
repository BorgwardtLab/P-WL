import numpy as np

class LabelGenerator():

    def __init__(self, start, length):
        self._current_label_index = 0
        self._start = start
        self._length = length
        self._label_generator = np.nditer(np.arange(start, start+length, 1))

    def get_next_label(self):
        if self._current_label_index < self._length:
            self._current_label_index += 1
            return str(next(self._label_generator))
        else:
            print("reset")
            self._label_generator = np.nditer(np.arange(self._start, self._start+self._length, 1)) 
            self._current_label_index = 1
            return str(next(self._label_generator))
            
