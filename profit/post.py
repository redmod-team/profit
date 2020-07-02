class Postprocessor:
    def __init__(self, interface):
        self.interface = interface
        self.eval_points = read_input()

    def read(self):
        nrun = self.eval_points.shape[1]
        cwd = os.getcwd()

        self.data = np.empty(np.append(self.interface.shape(), nrun))

        # TODO move this to UQ module
#        distribution = J(*uq.params.values())
#        nodes, weights = generate_quadrature(uq.backend.order + 1, distribution, rule='G')
#        expansion = orth_ttr(uq.backend.order, distribution)

        for krun in range(nrun):
            fulldir = path.join(config.run_dir, str(krun).zfill(3)) #.zfill(3) is an option that forces krun to have 3 digits$
            try:
                os.chdir(fulldir)
                self.data[:,krun] = self.interface.get_output()
            finally:
                os.chdir(cwd)

def evaluate_postprocessing(distribution,data,expansion):
    import matplotlib.pyplot as plt
    from profit import read_input
    from chaospy import generate_quadrature, orth_ttr, fit_quadrature, E, Std, descriptives

    nodes, weights = generate_quadrature(uq.backend.order+1, distribution, rule='G')
    expansion = orth_ttr(uq.backend.order, distribution)
    approx = fit_quadrature(expansion, nodes, weights, np.mean(data[:,0,:], axis=1))
    urange = list(uq.params.values())[0].range()
    vrange = list(uq.params.values())[1].range()
    u = np.linspace(urange[0], urange[1], 100)
    v = np.linspace(vrange[0], vrange[1], 100)
    U, V = np.meshgrid(u, v)
    c = approx(U,V)

    # for 3 parameters:
    #wrange = list(uq.params.values())[2].range()
    #w = np.linspace(wrange[0], wrange[1], 100)
    #W = 0.03*np.ones(U.shape)
    #c = approx(U,V,W)

    plt.figure()
    plt.contour(U, V, c, 20)
    plt.colorbar()
    plt.scatter(config.eval_points[0,:], config.eval_points[1,:], c = np.mean(data[:,0,:], axis=1))

    plt.show()

    F0 = E(approx, distribution)
    dF = Std(approx, distribution)
    sobol1 = descriptives.sensitivity.Sens_m(approx, distribution)
    sobolt = descriptives.sensitivity.Sens_t(approx, distribution)
    sobol2 = descriptives.sensitivity.Sens_m2(approx, distribution)

    print('F = {} +- {}%'.format(F0, 100*abs(dF/F0)))
    print('1st order sensitivity indices:\n {}'.format(sobol1))
    print('Total order sensitivity indices:\n {}'.format(sobolt))
    print('2nd order sensitivity indices:\n {}'.format(sobol2))
