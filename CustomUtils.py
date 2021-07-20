import os, sys, time
import numpy as np

def generator_getClassifiedItems(clf, generator, classified):
    temp_X = []
    temp_y = []
    for idx in range(generator.step_per_epoch):
        batch_x, batch_y = generator.__getitem__(idx)
        for i, val in enumerate(batch_x):
            c = clf.predict([val.flatten()])[0]
            if classified == c:
                temp_X.append(batch_x[i])
                temp_y.append(batch_y[i])
    return np.array(temp_X), np.array(temp_y)
    
def SaveSummary(model, path):
    with open(path + '/report.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

class ShowProcess():
    i = 0 
    max_steps = 0 
    max_arrow = 50 
    infoDone = 'Done'

    
    def __init__(self, max_steps, infoDone=None):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow 
        percent = self.i * 100.0 / self.max_steps 
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' 
        sys.stdout.write(process_bar) 
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        if self.infoDone:
            print(self.infoDone)
        self.i = 0

