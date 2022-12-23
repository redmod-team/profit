ntrain = 2
variables = {
    "u": "Halton()",
    "v": "Uniform(0.55, 0.6)",
    "w": "ActiveLearning()",
    "r": "Independent(0, 1, 10)",
    "f": {"kind": "Output", "dependent": ("r",)},
    "g": "Output(r)",
}

files = {"input": "input.txt", "output": "output.txt"}

run = {
    "pre": {"class": "template", "param_files": ["mockup.in"]},
    "post": {"class": "hdf5", "path": "mockup.out"},
    "command": "python3 mockup.py",
    "clean": False,
}
