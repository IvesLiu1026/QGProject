@echo off

for /L %%i in (78,1,80) do (
    echo Running test11.py with argument %%i
    python test11.py %%i
)