@echo off
rem ==========================================================================
rem  Analisis de marcha: doble clic para abrir el programa.
rem  Tambien se puede arrastrar un video sobre este archivo.
rem
rem  La primera vez instala Python (si falta) y las bibliotecas en un entorno
rem  propio, fuera de Google Drive: %LOCALAPPDATA%\AnalisisMarcha\venv
rem
rem  Opciones:  Analisis_Marcha.bat /instalar     solo instala
rem             Analisis_Marcha.bat /reinstalar   vuelve a instalar las bibliotecas
rem ==========================================================================
setlocal EnableExtensions
title Analisis de marcha
cd /d "%~dp0"

if not defined MARCHA_VENV set "MARCHA_VENV=%LOCALAPPDATA%\AnalisisMarcha\venv"
set "PY=%MARCHA_VENV%\Scripts\python.exe"
set "PYW=%MARCHA_VENV%\Scripts\pythonw.exe"
set "SELLO=%MARCHA_VENV%\requirements_instalado.txt"
set "ARG=%~1"

if /i "%ARG%"=="/reinstalar" goto :instalar
if not exist "%PY%" goto :instalar
rem Si requirements.txt cambio desde la ultima instalacion, se actualiza.
fc /b "requirements.txt" "%SELLO%" >nul 2>nul || goto :instalar
goto :abrir

:instalar
echo ============================================================
echo   ANALISIS DE MARCHA - preparando el programa
echo.
echo   Solo se hace la primera vez. Necesita internet y puede
echo   tardar de 5 a 15 minutos. No cierre esta ventana.
echo ============================================================
echo.
if exist "%PY%" goto :paquetes
call :buscar_python
if not defined PYBASE goto :sin_python
echo Python encontrado: %PYBASE%
echo Creando el entorno en %MARCHA_VENV% ...
"%PYBASE%" -m venv "%MARCHA_VENV%"
if errorlevel 1 goto :error

:paquetes
echo.
echo [1/3] Instalando las bibliotecas (OpenCV, NumPy, SciPy, pandas, ONNX Runtime, MediaPipe)...
"%PY%" -m pip install --disable-pip-version-check --upgrade pip
"%PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :error
echo.
echo [2/3] Instalando RTMPose (rtmlib)...
rem rtmlib pide opencv-python, que choca con opencv-contrib-python: se instala sin dependencias.
"%PY%" -m pip install --disable-pip-version-check --no-deps rtmlib==0.0.16
if errorlevel 1 goto :error
echo.
echo [3/3] Descargando los modelos de deteccion sin marcadores (unos 100 MB)...
"%PY%" -c "import tkinter, sin_marcadores as s; s.preparar_modelo('rtmpose'); s.preparar_modelo('mediapipe')"
if errorlevel 1 goto :error
copy /y "requirements.txt" "%SELLO%" >nul
echo.
echo Instalacion terminada.

:abrir
if /i "%ARG%"=="/instalar" exit /b 0
if /i "%ARG%"=="/reinstalar" set "ARG="
if not exist "%PYW%" goto :error
if defined ARG (start "" "%PYW%" "%~dp0lanzador.py" "%ARG%") else start "" "%PYW%" "%~dp0lanzador.py"
exit /b 0

:sin_python
echo.
echo No se encontro Python 3.10 o mas reciente de 64 bits.
where winget >nul 2>nul || goto :sin_winget
choice /c SN /m "Desea instalar Python 3.12 automaticamente"
if errorlevel 2 goto :sin_winget
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
call :buscar_python
if not defined PYBASE goto :sin_winget
goto :instalar

:sin_winget
echo.
echo Instale Python desde https://www.python.org/downloads/
echo (en el instalador marque "Add python.exe to PATH")
echo y vuelva a abrir este archivo.
start "" "https://www.python.org/downloads/"
pause
exit /b 1

:error
echo.
echo *** Ocurrio un error. ***
echo Revise su conexion a internet y vuelva a abrir este archivo.
echo Si el error continua, ejecute "Analisis_Marcha.bat /reinstalar"
echo o envie una captura de esta ventana.
pause
exit /b 1

rem --------------------------------------------------------------------------
:buscar_python
set "PYBASE="
for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PYBASE=%%i"
call :verificar_python
if defined PYBASE exit /b 0
for /f "delims=" %%i in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PYBASE=%%i"
call :verificar_python
if defined PYBASE exit /b 0
for %%v in (313 312 311 310) do if not defined PYBASE if exist "%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe" set "PYBASE=%LOCALAPPDATA%\Programs\Python\Python%%v\python.exe"
call :verificar_python
exit /b 0

:verificar_python
if not defined PYBASE exit /b 0
"%PYBASE%" -c "import sys,struct; sys.exit(0 if sys.version_info >= (3, 10) and struct.calcsize('P') == 8 else 1)" >nul 2>nul || set "PYBASE="
exit /b 0
