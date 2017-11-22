from keras.layers import *
from keras.models import *
import pycbc.psd
import gmind.generator

p = pycbc.psd.aLIGOZeroDetHighPower(2**19, 1.0/16, 15)
s = gmind.generator.GaussianNoiseGenerator(32, 1024, p, 20)
p = gmind.generator.WFParamGenerator(["examples/test.ini"])
g = w = gmind.generator.GaussianSignalTimeGenerator(s, p, [-3, 1], batch=100)

# dimensions of our images.
width = 1024 * 4
 
model = Sequential()
model.add(Conv1D(32, 3, input_shape=(width, 1), padding='causal'))
model.add(BatchNormalization())

for i in range(6):
    model.add(Conv1D(32, 8, padding='causal'))
    model.add(BatchNormalization())

for i in range(4):
    model.add(Conv1D(32, 3, padding='causal'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("linear"))
model.add(Dense(16))
model.add(Dense(1))

test = False
retrain = False
if test or retrain:
    model.load_weights('test.hdf')

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

if test:
    s, t = g.next()
    p = model.predict(s, 1)
    import pylab
    pylab.figure(1)
    pylab.scatter(t, p)
    pylab.show()

if not test:
    model.fit_generator(g, 20, epochs=1)
    model.save('test.hdf')

