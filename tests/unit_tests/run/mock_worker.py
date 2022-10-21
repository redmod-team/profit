import profit

from profit.run import Worker


@Worker.wrap("mock")
def work(u, v) -> "f":
    return u + v
