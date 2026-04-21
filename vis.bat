@echo off
REM vis.bat -- Build via build.bat, then launch the exe.

setlocal
pushd "%~dp0"

call build.bat
if errorlevel 1 (
    popd
    exit /b 1
)

set EXE=build\Release\gauss_map_viz.exe
if not exist "%EXE%" set EXE=build\gauss_map_viz.exe
if not exist "%EXE%" (
    echo Could not find built gauss_map_viz.exe under build\
    popd
    exit /b 1
)

echo Launching %EXE% ...
start "" "%EXE%"
popd
endlocal
