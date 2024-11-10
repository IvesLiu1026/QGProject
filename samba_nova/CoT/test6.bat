@echo off

for /L %%i in (43,1,44) do (
    echo Running test6.py with argument %%i
    python test6.py %%i
)