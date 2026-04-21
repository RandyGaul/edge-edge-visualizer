@echo off
REM build.bat -- Configure and build via CMake (default generator).

setlocal
pushd "%~dp0"
set BUILD_DIR=build

if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Configuring CMake project...
    cmake -S . -B "%BUILD_DIR%"
    if errorlevel 1 (
        echo CMake configure FAILED.
        popd
        exit /b 1
    )
)

echo Building (Release)...
cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 (
    echo Build FAILED.
    popd
    exit /b 1
)

echo Build OK.
popd
endlocal
