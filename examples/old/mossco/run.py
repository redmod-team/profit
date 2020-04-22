import profit
from profit.run.backend import LocalCommand

runner = profit.Runner(LocalCommand('./template/sediment_io'))

runner.start()
