@echo off
title YOLO-Master A2 baseline - GPU memory capped at 6 GiB
cd /d "C:\Users\yqy\.codex\worktrees\5cf3\YOLO-Master"
powershell -NoProfile -Command "& 'C:\Users\yqy\.conda\envs\yolo_master\python.exe' 'A2_OfficialReproduce\run_baseline_gpu6g.py' 2>&1 | Tee-Object -FilePath 'A2_OfficialReproduce\baseline_gpu6g_console.log'"
pause
