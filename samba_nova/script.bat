@echo off
for /L %%i in (5,1,122) do (
    echo Running main.py with argument %%i
    python main.py %%i
)
