This is a mockup example case for proFit.

Two ways of running proFit:

1) Using the console scripts based on config files
2) Using the Python API

Let's assume we have a simulation code that is run by calling 

```
python mockup.py 
```

Here, `mockup.py` requires two scalar input parameters: `u` and `v` and returns a scalar result `f(u,v)`.
The input is read from an input file `mockup.in`. After computation, the output is saved to `mockup.out`.

## Parameter study via console scripts
We want to do a parameter study by varying `u` and `v`. First we enter the `study` directory and
edit the file `profit.yml`. Then we generate run directories with
```
profit pre
```
This will generate run directories `run` where `u` and `v` are replaced by quasi-random values
inside the domain of interest. In addition a file `input.txt` containing all parameter 
combinations for runs is generated.

Now we can start runs with
```
profit run
```

Results from run folders are collected with
```
profit collect
```
yielding a file `output.txt` with the entries from `run` directories.
