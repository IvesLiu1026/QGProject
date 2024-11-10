@echo off

for /L %%i in (92,1,98) do (
    echo Running test13.py with argument %%i
    python test13.py %%i
)