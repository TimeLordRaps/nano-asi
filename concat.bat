@echo off
setlocal enabledelayedexpansion

REM Output file
set output_file=combined_code.md

REM Delete the output file if it exists
if exist %output_file% del %output_file%

REM Define core module paths
set "core_modules=nano_asi\modules\consciousness\tracker.py nano_asi\modules\evaluation\benchmarks.py nano_asi\modules\judgement\base.py nano_asi\modules\judgement\criteria.py nano_asi\modules\judgement\strategies.py nano_asi\modules\mcts.py nano_asi\modules\mcts\advanced.py nano_asi\modules\mcts\base.py nano_asi\modules\mcts\nodes.py nano_asi\modules\mcts\states.py"

REM Loop through core module paths
for %%f in (%core_modules%) do (
    echo Adding file: %%f
    REM Add filename separator
    echo ================================================== >> %output_file%
    echo File: %%f >> %output_file%
    echo ================================================== >> %output_file%
    REM Add file content
    type "%%f" >> %output_file%
    echo. >> %output_file%
)

echo Core nano-asi modules concatenated into %output_file%.
