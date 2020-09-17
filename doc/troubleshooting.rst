Troubleshooting
===============

Problem: Kernel initialization on Mac

Solution: Everywhere you see `python`on the documentation, `python3`have to be used instead.
If it still doesn't work, it's probably due to the fact that numpy, which contains f2py (fortran to python), isn't correctly installed.
Usualy it's better to use 'brew' when it's available and pip for the rest
`brew install numpy`
`brew install scipy`
`brew install sympy`

Now Fortran compiler should be installed, you can verify it with
`$gfortran`
which return a fatal error.



Problem: command :code:`profit` not found

Solution: The path of profit isn't in the system path.

Make sure that you use Python 3.8, if it's not the case upgrade it through

'brew upgrade python'

Find the path with
'$ls /usr/local/opt/python@3.8/bin'

Create the file ".zhrc" if it doesn't exist with
`$vim .zshrc`

`$source ~/.bashrc`
`$source ~/.zshrc `

Download again profit by using python3 (pip3)

`pip3 install -e . --user `

