@echo off
for /L %%i in (63 80 111 120) do (
    echo Running main3.py with argument %%i
    python main3.py %%i
)

:: 10 13 17 20 22 23 26 28 36 37 39 40 41 42 43 44 45 46 49 50 51 59 62 63 80 84 96 109 110 111 120
