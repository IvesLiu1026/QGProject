@echo off
:: run with argument 80 83 84 98 99
:: use the list of numbers in the for loop


for %%i in (33 34 44) do (
    echo Running main.py with argument %%i
    python main.py %%i
)