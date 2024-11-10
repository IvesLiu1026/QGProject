@echo off

for /L %%i in (1,1,7) do (
    echo Running test0.py with argument %%i
    python test0.py %%i
)