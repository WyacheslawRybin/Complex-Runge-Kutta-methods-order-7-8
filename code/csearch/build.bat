@echo off
rem build.bat CFGDIR MH EXE   e.g.  build.bat cfg77 6 casc77_6.exe
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
if exist %3 del %3
cl /nologo /O2 /fp:fast /arch:AVX2 /openmp /I %1 /DMH=%2 /Fe:%3 cascsearch.c /Fo:tmp_%3.obj
