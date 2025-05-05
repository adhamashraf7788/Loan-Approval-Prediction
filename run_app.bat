@echo off
echo Starting Flask Application for ML-Project...

REM Activate your virtual environment (uncomment and update if using one)
REM call C:\Users\co.magic\Desktop\mlll\ML-Project\venv\Scripts\activate

REM Change to your project directory
cd /d C:\Users\co.magic\Desktop\mlll\ML-Project

REM Run the Flask app in the background
start /b python app.py

REM Wait 2 seconds for the server to start
timeout /t 2 /nobreak

REM Open the default browser
start http://localhost:5000

exit