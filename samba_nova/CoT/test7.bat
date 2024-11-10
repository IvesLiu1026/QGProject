@echo off

for /L %%i in (50,1,56) do (
    echo Running test7.py with argument %%i
    python test7.py %%i
)