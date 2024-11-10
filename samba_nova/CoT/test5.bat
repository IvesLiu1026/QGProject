@echo off

for /L %%i in (36,1,42) do (
    echo Running test5.py with argument %%i
    python test5.py %%i
)