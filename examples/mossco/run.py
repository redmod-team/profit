import suruq
from suruq.run.backend import LocalCommand

runner = suruq.Runner(LocalCommand('./template/sediment_io'))

runner.start()
