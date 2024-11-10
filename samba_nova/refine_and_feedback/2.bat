@echo off

for /L %%i in (25,1,48) do (
    echo Running 2.py with argument %%i
    python 2.py %%i
)