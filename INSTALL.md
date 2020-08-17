## To use fortran for the kernels:

## Linux:
In the terminal run each one of the following commands: 
```bash
python3 init_func.py
f2py -c kernels.f90 -m kernels
```
	
## Windows:
*Only the first time:*

Donwload [chocolatey](https://chocolatey.org/install):

1. On the windows powershell executed with admin rights, type each of the following commands: 
```
Get-ExecutionPolicy
```

if it returns 'Restricted' type the command: 
```
Set-ExecutionPolicy AllSigned
```

To continue the installation run:
```
Get-ExecutionPolicy
Set-ExecutionPolicy Bypass -Scope Process
choco
```

2. On the Windows powershell executed without admin rights type the following command: 
```
choco install mingw
choco install git
```
					
3. Open git-bash as Administrator:
```
echo -e "[build]\ncompiler = mingw32" > /c/tools/Anaconda3/Lib/distutils/distutils.cfg
```

4. On the Anaconda-prompt type this command: 
```
gfortran -v
```
          
*Always:*

On the Anaconda-prompt:
```
python init_func.py
f2py -c kernels.f90 -m kernels 
```
