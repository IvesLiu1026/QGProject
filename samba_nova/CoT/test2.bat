@echo off

for /L %%i in (15,1,16) do (
    echo Running test2.py with argument %%i
    python test2.py %%i
)