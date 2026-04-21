@echo off
REM Build gauss_map_viz.c
REM Tries MSVC first, falls back to MinGW-w64 gcc.

where cl >nul 2>&1
if %errorlevel%==0 (
    echo Building with MSVC...
    cl /nologo /O2 gauss_map_viz.c opengl32.lib glu32.lib gdi32.lib user32.lib
) else (
    where gcc >nul 2>&1
    if %errorlevel%==0 (
        echo Building with GCC...
        gcc -O2 gauss_map_viz.c -o gauss_map_viz.exe -lopengl32 -lglu32 -lgdi32 -luser32 -mwindows
    ) else (
        echo ERROR: No compiler found. Install MSVC or MinGW-w64.
        exit /b 1
    )
)

if %errorlevel%==0 (
    echo Build OK: gauss_map_viz.exe
) else (
    echo Build FAILED.
)
