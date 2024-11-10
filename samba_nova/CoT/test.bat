@echo off
setlocal EnableDelayedExpansion

for /L %%N in (0,1,16) do (
    set /A Start=1 + %%N * 7
    if %%N LSS 16 (
        set /A End=!Start! + 6
    ) else (
        set End=122
    )
    (
        echo @echo off
        echo.
        echo for /L %%%%i in (!Start!,1,!End!) do (
        echo     echo Running %%N.py with argument %%%%i
        echo     python %%N.py %%%%i
        echo )
    ) > %%N.bat
)
