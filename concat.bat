@echo off
setlocal enabledelayedexpansion

REM Output file
set output_file=combined_code.md

REM Delete the output file if it exists
if exist %output_file% del %output_file%

REM Loop through all relevant files
for /r %%f in (*.py) do (
    REM Get the filename without the directory
    set filename=%%~nxf
    
    REM Exclude __init__.py and files starting with a dot
    if not "!filename!"=="__init__.py" (
        if not "!filename:~0,1!"=="." (
            echo Adding file: %%~f
            REM Add filename separator
            echo ================================================== >> %output_file%
            echo File: %%~f >> %output_file%
            echo ================================================== >> %output_file%
            REM Add file content
            type "%%f" >> %output_file%
            echo. >> %output_file%
        )
    )
)

echo All files concatenated into %output_file%.
