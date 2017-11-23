from keras.layers import *
from keras.models import *
import pycbc.psd
import gmind.generator
import matplotlib
matplotlib.use('Agg')

p = pycbc.psd.aLIGOZeroDetHighPower(2**19, 1.0/16, 15)
s = gmind.generator.GaussianNoiseGenerator(32, 1024, p, 20)
p = gmind.generator.WFParamGenerator(["examples/test.ini"])
g = w = gmind.generator.GaussianSignalTimeGenerator(s, p, [-3, 1], batch=50)

# dimensions of our images.
width = 1024 * 4
 
model = Sequential()
model.add(Conv1D(32, 3, input_shape=(width, 1), padding='causal'))

for i in range(6):
    model.add(Conv1D(32, 8, padding='causal'))

for i in range(6):
    model.add(Conv1D(32, 3, padding='causal'))
    model.add(MaxPooling1D(2))

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(16))
model.add(Dense(1))

retrain = True
if retrain:
    model.load_weights('test.hdf')

model.compile(loss='mean_squared_error',
              optimizer='nadam',
              metrics=['accuracy'])

i = 0
while 1:
    #FIT ME
    d, t = g.next()
    model.fit(d, t, batch_size=g.batch, epochs=100)
    model.save('test.hdf')
    
    #TESTS I UNDERSTAND
    s, t = g.next()
    p = model.predict(s, 1)
    
    ms = (t - p)**2.0
    print i, ms.mean(), ms.max(), p.min(), p.max()
    
    import pylab
    pylab.figure()
    pylab.scatter(t, p)
    pylab.xlim(4, 20)
    pylab.ylim(4, 20)
    pylab.savefig('iter-%s.png' % i)
    pylab.close('all')
    i += 1

