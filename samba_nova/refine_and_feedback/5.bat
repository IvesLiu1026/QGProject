@echo off

for /L %%i in (97,1,122) do (
    echo Running 5.py with argument %%i
    python 5.py %%i
)