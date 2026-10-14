@echo off
rem build_bench.bat  -> bench.exe (MSVC 2022, /O2, as in the 5-6 paper)
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
if exist bench.exe del bench.exe
cl /nologo /O2 /EHsc /std:c++17 bench.cpp /Fe:bench.exe /Fo:bench.obj
