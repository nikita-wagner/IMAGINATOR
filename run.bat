@echo off
rem Run AI Image Studio from source using the project virtual environment.
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
	echo Could not find .venv\Scripts\pythonw.exe
	echo Run this script from inside the AI-text-to-img folder, after the
	echo virtual environment has been set up.
	pause
	exit /b 1
)

rem pythonw.exe has no console, so a crash before the window even appears
rem would otherwise be invisible - main.py catches that case and writes
rem crash.log next to this script.
start "AI Image Studio" /D "%~dp0" ".venv\Scripts\pythonw.exe" main.py
