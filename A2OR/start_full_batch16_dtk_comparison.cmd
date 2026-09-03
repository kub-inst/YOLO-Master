@echo off
set "YOLO_CONFIG_DIR=D:\coding\YOLO-Master\A2OR\.ultralytics_config"
cd /d D:\coding\YOLO-Master
conda run --no-capture-output -n yolo_master python A2OR\run_full_batch16_dtk_comparison.py
echo.
echo Full-data comparison exited with code %ERRORLEVEL%
