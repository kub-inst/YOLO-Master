@echo off
title A2OR v0.1-N DTK lambda=0.8 VisDrone10%% 20e w1
cd /d "D:\coding\YOLO-Master"
"C:\Users\yqy\.conda\envs\yolo_master\python.exe" -u "A2OR\run_dtk_v01n_visdrone10_s42_20e_w1.py"
set EXIT_CODE=%ERRORLEVEL%
echo.
echo A2OR DTK process exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
