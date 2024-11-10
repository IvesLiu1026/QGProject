@echo off

for /L %%i in (85,1,91) do (
    echo Running test12.py with argument %%i
    python test12.py %%i
)