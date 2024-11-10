@echo off

for /L %%i in (57,1,63) do (
    echo Running test8.py with argument %%i
    python test8.py %%i
)