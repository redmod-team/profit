import profit 
from profit.run.backend import LocalCommand

runner = profit.Runner(LocalCommand('python mockup.py'))
runner.start()
