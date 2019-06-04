import suruq
from suruq.run.backend import LocalCommand

runner = suruq.Runner(LocalCommand('python mockup.py'))
runner.start()
