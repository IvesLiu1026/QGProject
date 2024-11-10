@echo off

for %%i in (29 30 35) do (
    echo Running test4.py with argument %%i
    python test4.py %%i
)