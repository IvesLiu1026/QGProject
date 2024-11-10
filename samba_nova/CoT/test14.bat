@echo off

for /L %%i in (99,1,105) do (
    echo Running test14.py with argument %%i
    python test14.py %%i
)