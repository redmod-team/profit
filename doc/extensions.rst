.. _extensions:

Custom extensions
=================

Experienced users have the possibility to add custom components which are used
throughout the proFit workflow. The code should be placed in a Python file which
is referenced inside the ``include`` section of the ``profit.yaml`` configuration file.

Following components are customizable:

* Runner
    | Base class: ``profit.run.Runner``
    | Set in ``run`` config: ``runner: label_of_custom_runner``
* RunnerInterface
    | Base class: ``profit.run.RunnerInterface``
    | Set in ``run`` config: ``interface: label_of_custom_interface``
* Worker
    | Base class: ``profit.run.Worker``
    | Set in ``run`` config: ``worker: label_of_custom_worker``
* Preprocessor
    | Base class: ``profit.run.Preprocessor``
    | Set in ``run`` config: ``preprocessor: label_of_custom_preprocessor``
* Postprocessor
    | Base class: ``profit.run.Postprocessor``
    | Set in ``run`` config: ``postprocessor: label_of_custom_postprocessor``
* Surrogate model
    | Base class: ``profit.sur.Surrogate``
    | Set in ``fit`` config: ``surrogate: label_of_custom_surrogate``
* Active learning algorithm
    | Base class: ``profit.al.ActiveLearning``
    | Set in ``active_learning`` config: ``algorithm: label_of_custom_al_algorithm``
* Acquisition function
    | Base class: ``profit.al.acquisition_functions.AcquisitionFunction``
    | Set in ``active_learning/algorithm`` config: ``acquisition_function: label_of_custom_acquisition_function``
* FileHandler in-/output file format
    | Base class: ``profit.util.file_handler.FileHandler``
    | Set in ``files`` config: File ending of custom ``FileHandler``

To create custom classes, the method ``register`` of the corresponding base
class is used. For the ``Worker``, ``Preprocessor`` and ``Postprocessor`` classes there exists a ``wrap`` method which simplifies the registering process.
This method is planned also for the other components.

Examples
--------

Here, examples of registering a custom worker, custom postprocessor and a custom file format are shown.

.. code-block:: python

    # Worker

    from profit.run import Worker
    import numpy as np


    @Worker.register('custom_worker1')
    class CustomWorker(Worker):
    """Directly calling the wanted python function."""

        def main(self):
            u = self.interface.input['u']
            v = self.interface.input['v']
            self.interface.output['f'] = np.cos(10 * u) + v
            self.interface.done()


    @Worker.register('custom_worker2')
    class CustomWorker(Worker):
    """Substituting the run method to do something special, pre and post is executed as usual."""

        def run(self):
            params = np.loadtxt('mockup.in')
            u = params[0]
            v = params[1]
            f = np.cos(10 * u) + v
            np.savetxt('mockup.out', np.array([f]))


    @Worker.wrap("custom_worker3", [u, v], f)
    def f(u, v):
    """Shorthand for custom_worker."""
        return np.cos(10 * u) + v

.. code-block:: python

    # Postprocessor

    from profit.run import Postprocessor
    import numpy as np


    @Postprocessor.register('custom_postprocessor1')
    class CustomPost(Postprocessor):
    """Almost identical copy of NumpytxtPostprocessor."""

        def post(self, data):
            raw = np.loadtxt('mockup.out')
            data['f'] = raw


    @Postprocessor.wrap('custom_postprocessor2')
    def post(data):
        """Shorthand for custom_postprocessor."""
        raw = np.loadtxt('mockup.out')
        data['f'] = raw

.. code-block:: python

    # FileHandler in-/output file format

    from profit.util.file_handler import FileHandler


    @FileHandler.register("pkl")
    class PickleHandler(FileHandler):

    @classmethod
    def save(cls, filename, data, **kwargs):
        from pickle import dump
        write_method = 'wb' if not 'method' in kwargs else kwargs['method']
        dump(data, open(filename, write_method))

    @classmethod
    def load(cls, filename, as_type='raw', read_method='rb'):
        from pickle import load
        if as_type != 'raw':
            return NotImplemented
        return load(open(filename, read_method))
