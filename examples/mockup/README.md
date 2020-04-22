This is a mockup example case for proFit.

Two ways of running proFit:

1) Using the console scripts based on config files
2) Using the Python API

Let's assume we have a simulation code that is run by
calling 

```
python mockup.py 
```

Here, `mockup.py` requires two scalar input parameters: `u` and `v` and returns a 1D array.
The input is read from an input file `mockup.in`. After computation, the output is saved to `mockup.out`.
