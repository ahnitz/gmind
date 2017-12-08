from keras.layers import *
from keras.models import *
import pycbc.psd
import gmind.generator
import matplotlib
matplotlib.use('Agg')
import pylab, numpy

class flow_from_file(object):
    def __init__(self, fname):
        self.file = h5py.File(fname, 'r')
        self.data = self.file['data']
        self.targets = self.file['target']
        self.i = 0
        self.num_batches = len(self.data)

    def next(self):
        print self.i, self.num_batches
        d = self.data[str(self.i)][:]
        t = self.targets[str(self.i)][:]
        self.i += 1
        if self.i == self.num_batches:
            self.i = 0

        return d, t
g = flow_from_file('pregen.hdf')

# dimensions of our images.
width = 1024 * 4
 
model = Sequential()
model.add(Conv1D(32, 3, input_shape=(width, 1), padding='causal'))

for i in range(1):
    model.add(Conv1D(32, 3, padding='causal'))

for i in range(5):
    model.add(Conv1D(32, 3, padding='causal'))
    model.add(MaxPooling1D(2))

model.add(Flatten())
model.add(Dense(32))
model.add(Dense(1))

retrain = False
if retrain:
    model.load_weights('test.hdf')

model.compile(loss='mean_squared_error',
              optimizer='nadam',
              metrics=['accuracy'])

i = 0
d, t = g.next()
while 1:
    model.fit_generator(g, steps_per_epoch=g.num_batches, epochs=1)
    model.save('test.hdf')

    p = model.predict(d)

    pylab.figure()
    pylab.scatter(t, p, label='Testing')
    pylab.legend()
    pylab.xlim(1, 25)
    pylab.ylim(1, 25)
    pylab.savefig('iter-%s.png' % i)
    pylab.close('all')
    i += 1
