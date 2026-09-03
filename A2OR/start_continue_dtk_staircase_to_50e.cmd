@echo off
set "YOLO_CONFIG_DIR=D:\coding\YOLO-Master\A2OR\.ultralytics_config"
cd /d D:\coding\YOLO-Master
conda run --no-capture-output -n yolo_master python A2OR\continue_dtk_staircase_to_50e.py >> A2OR\continue_dtk_staircase_to_50e_console.log 2>&1
echo.
echo Continuation process exited with code %ERRORLEVEL%
