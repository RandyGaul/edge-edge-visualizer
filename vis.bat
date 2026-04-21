@echo off
REM vis.bat -- Configure (VS2019, x64), build Release via CMake + MSVC, launch.

setlocal
set ROOT=%~dp0
set BUILD_DIR=%ROOT%build

if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Configuring CMake project [Visual Studio 16 2019, x64]...
    cmake -S "%ROOT%" -B "%BUILD_DIR%" -G "Visual Studio 16 2019" -A x64
    if errorlevel 1 (
        echo CMake configure FAILED.
        exit /b 1
    )
)

echo Building (Release) with MSVC...
cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 (
    echo Build FAILED.
    exit /b 1
)

set EXE=%BUILD_DIR%\Release\gauss_map_viz.exe
if not exist "%EXE%" (
    echo Expected %EXE% but it was not produced.
    exit /b 1
)

echo Launching %EXE% ...
start "" "%EXE%"
endlocal
