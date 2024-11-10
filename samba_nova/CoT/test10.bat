@echo off

for /L %%i in (71,1,77) do (
    echo Running test10.py with argument %%i
    python test10.py %%i
)