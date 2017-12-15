import pycbc.psd
import gmind.generator
import pylab, numpy, h5py


total=2000
bsize = 200

p = pycbc.psd.aLIGOZeroDetHighPower(2**19, 1.0/16, 15)
s = gmind.generator.GaussianNoiseGenerator(32, 1024, p, 20)
p = gmind.generator.WFParamGenerator(["test.ini"])
g = gmind.generator.OWGaussianSignalTimeGenerator(s, p, [-3, 1], batch=bsize)
g.target_type = ['rsnr', 'isnr', 'time', 'mass1', 'mass2']

f = h5py.File('./pregen.hdf', 'w')

for i in range(total/bsize):
    d, t = g.next()
    f['data/%s' % i] = d
    f['target/%s' % i] = t

    params = g.current_params
    for k in params[0]:
        data = numpy.array([p[k] for p in params])
        f['param/%s/%s' % (i, k)] = data
    print "Test print of data %s", i

f.close()

