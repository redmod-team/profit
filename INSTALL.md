## To use fortran for the kernels:

## Linux:
In the terminal run each one of the following commands: 
```bash
git clone https://github.com/redmod-team/profit.git

cd profit
python3 -m venv venv
source venv/bin/activate

pip install -e .
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

## Windows via WSL2:
1. Install WSL2 on Windows 10 (Ubuntu 20.04 LTS)
    1. (optional) install PyCharm for Windows (tested with Professional)
    2. adjusting PyCharm according to the [documentation](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html)
   
2. [WSL] clone git-repository 
      
   important: the project should be located in the linux subsystem not the windows system
   
        git clone https://github.com/redmod-team/profit.git

3. [WSL] prepare install
   
         sudo apt-get install gfortran make python3-dev
         pip3 install -r requirements.txt
         
3. [WSL] install proFit (development-mode, with pip / setup.py)

         cd profit
         pip3 install -e .
         python3 setup.py build
   
   1. [WSL] install missing dependencies as required with pip
   
      h5py
      dash
      pandas
   
4. [WSL] fixed `PATH`: add the following line to your `.bashrc`
    
        export PATH=$HOME/.local/bin:$PATH
    
5. [WSL] profit should be recognized now


tested on 2021-02-12