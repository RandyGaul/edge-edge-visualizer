@echo off
REM web.cmd -- Configure and build for web via Emscripten (../emsdk).

setlocal
pushd "%~dp0"
set EMSDK_DIR=%~dp0..\emsdk
set EM_DIR=%EMSDK_DIR%\upstream\emscripten
set BUILD_DIR=build_web

if not exist "%EMSDK_DIR%\emsdk_env.bat" (
    echo Could not find emsdk at "%EMSDK_DIR%".
    popd
    exit /b 1
)

echo Activating emsdk...
call "%EMSDK_DIR%\emsdk_env.bat" >nul 2>&1

REM Emscripten needs ninja or make. Pick up ninja from Visual Studio if present.
set "VS_NINJA=C:\Program Files\Microsoft Visual Studio\18\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if exist "%VS_NINJA%\ninja.exe" set "PATH=%VS_NINJA%;%PATH%"

REM The emsdk env script may not propagate PATH correctly under some shells,
REM so invoke emcmake.bat directly via its full path.
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Configuring with emcmake...
    call "%EM_DIR%\emcmake.bat" cmake -S . -B "%BUILD_DIR%" -G Ninja
    if errorlevel 1 (
        echo emcmake configure FAILED.
        popd
        exit /b 1
    )
)

echo Building (Release)...
call cmake --build "%BUILD_DIR%"
if errorlevel 1 (
    echo Build FAILED.
    popd
    exit /b 1
)

echo Build OK. Output in %BUILD_DIR%\
popd
endlocal
