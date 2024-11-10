@echo off

for /L %%i in (113,1,122) do (
    echo Running test16.py with argument %%i
    python test16.py %%i
)