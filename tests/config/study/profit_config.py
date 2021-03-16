ntrain = 2
variables = {'u': 'Halton()',
             'v': 'Uniform(0.55, 0.6)',
             'w': 'ActiveLearning()',
             'r': 'Independent(0, 1, 0.1)',
             'f': {'kind': 'Output', 'depend': ('r',), 'range': 'r'},
             'g': 'Output(r)'}

files = {'input': 'input.txt',
         'output': 'output.txt'}

run = {'pre': {'class': 'template', 'param_files': ['mockup.in']},
       'post': {'class': 'numpytxt', 'path': 'mockup.out'},
       'command': 'python3 mockup.py',
       'clean': False}
