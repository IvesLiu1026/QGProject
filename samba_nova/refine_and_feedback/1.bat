@echo off

for /L %%i in (1,1,24) do (
    echo Running 1.py with argument %%i
    python 1.py %%i
)