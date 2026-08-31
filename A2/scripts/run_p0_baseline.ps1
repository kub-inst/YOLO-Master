$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\coding\YOLO-Master"

Write-Host "Starting P0: YOLO26n + VisDrone default TAL/STAL, 50 epochs" -ForegroundColor Cyan
Write-Host "Checkpoints: every epoch; assignment telemetry: enabled" -ForegroundColor Cyan

conda run --no-capture-output -n yolo_master yolo detect train `
    model="D:/coding/YOLO-Master/yolo26n.pt" `
    data="D:/coding/YOLO-Master/A2/configs/visdrone.yaml" `
    epochs=50 imgsz=640 batch=2 device=0 workers=0 `
    seed=42 deterministic=True `
    optimizer=AdamW lr0=0.001 lrf=0.01 momentum=0.937 weight_decay=0.0005 `
    warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 `
    mosaic=1.0 mixup=0.1 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 `
    close_mosaic=10 patience=50 `
    assignment_stats=True assignment_small_area=1024.0 assignment_medium_area=9216.0 `
    save=True save_period=1 save_json=True val=True plots=True `
    project="D:/coding/YOLO-Master/A2/runs" name="p0_y26n_vd640_s42_50e" exist_ok=False

Write-Host "Training process exited with code $LASTEXITCODE" -ForegroundColor Yellow
