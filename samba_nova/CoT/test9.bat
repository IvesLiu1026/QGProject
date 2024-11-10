@echo off

for /L %%i in (64,1,70) do (
    echo Running test9.py with argument %%i
    python test9.py %%i
)