@echo off
cd /d "%~dp0"
echo Descent Lab: http://localhost:8000
py -m http.server 8000
