@echo off

for /L %%i in (106,1,112) do (
    echo Running test15.py with argument %%i
    python test15.py %%i
)