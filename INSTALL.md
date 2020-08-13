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
					```bash
          Get-ExecutionPolicy
						if it returns 'Restricted' type the command: Set-ExecutionPolicy AllSigned
	 				Get-ExecutionPolicy
					Set-ExecutionPolicy Bypass -Scope Process
	 				choco
          ```

      2. On the Windows powershell executed without admin rights type the following command: 
          ```bash
					choco install mingw
					choco install git
          ```
					
      3. Open git-bash as Administrator:
          ```bash
					echo -e "[build]\ncompiler = mingw32" > /c/tools/Anaconda3/Lib/distutils/distutils.cfg
          ```

      4. On the Anaconda-prompt type this command: 
          ```bash
					gfortran -v
          ```
          
*Always:*
On the Anaconda-prompt:
      ```bash
			python init_func.py
			f2py -c kernels.f90 -m kernels 
      ```
