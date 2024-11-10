@echo off

for /L %%i in (22,1,28) do (
    echo Running test3.py with argument %%i
    python test3.py %%i
)