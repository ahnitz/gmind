from keras.applications.mobilenet import MobileNet
import pycbc.psd
import gmind.generator

p = pycbc.psd.aLIGOZeroDetHighPower(2**19, 1.0/16, 15)
s = gmind.generator.GaussianNoiseGenerator(16, 1024, p, 20)
p = gmind.generator.WFParamGenerator(["examples/test.ini"])
w = gmind.generator.GaussianSignalQImageGenerator(s, p, 3, (224, 224), q=20)

i, t = w.next()
print i.shape, t
model = MobileNet(classes=1, weights=None)
model.compile(loss='mean_squared_error',
              optimizer='adagrad',
              metrics=['accuracy'])

model.fit_generator(w, 100, epochs=1)
