@echo off

for %%i in (49 51 52 53 58 60) do (
    echo Running 3.py with argument %%i
    python 3.py %%i
)

:: 49 51 52 53 58 60 