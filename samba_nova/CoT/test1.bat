@echo off

for /L %%i in (9,1,12) do (
    echo Running test1.py with argument %%i
    python test1.py %%i
)