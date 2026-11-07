@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Python virtual environment not found.
    echo Expected: "%PYTHON%"
    echo Create it with: py -m venv .venv
    pause
    exit /b 1
)

:menu
echo.
echo RAG Project
echo ----------
echo 1. Start website
echo 2. Start RAG question answering
echo 3. Exit
set "choice="
set /p "choice=Choose an option: "

if "%choice%"=="1" goto website
if "%choice%"=="2" goto rag
if "%choice%"=="3" exit /b 0
echo Invalid choice.
goto menu

:website
if not exist "%~dp0lets-scroll-main\index.html" (
    echo ERROR: Website files were not found.
    pause
    goto menu
)

echo Starting website at http://127.0.0.1:8000
echo Close the server window to stop the website.
start "RAG Project Website Server" /D "%~dp0lets-scroll-main" "%PYTHON%" -m http.server 8000 --bind 127.0.0.1
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8000/"
goto menu

:rag
if not exist "%~dp0chroma_db\chroma.sqlite3" (
    echo ERROR: Chroma database is missing.
    echo Restore or build the database before starting question answering.
    pause
    goto menu
)

echo Starting interactive RAG question answering...
"%PYTHON%" "%~dp0predict.py"
if errorlevel 1 (
    echo.
    echo RAG question answering exited with an error.
    pause
)
goto menu
