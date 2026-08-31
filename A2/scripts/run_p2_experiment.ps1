$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\coding\YOLO-Master"

Write-Host "Starting P2 Experiment: YOLO26n-P2 + VisDrone, P2-gated (small targets only), 20 epochs" -ForegroundColor Cyan
Write-Host "Model: yolo26n-p2.yaml (P2/4 + P3/8 + P4/16 + P5/32)" -ForegroundColor Cyan
Write-Host "P2 Gating: only small targets (area < 1024) use stride=4 anchors" -ForegroundColor Cyan
Write-Host "Expansion: <8px -> 16px (unchanged from baseline)" -ForegroundColor Cyan

conda run --no-capture-output -n yolo_master yolo detect train `
    model="yolo26n-p2.yaml" `
    data="D:/coding/YOLO-Master/A2/configs/visdrone.yaml" `
    epochs=20 imgsz=640 batch=2 device=0 workers=0 `
    seed=42 deterministic=True `
    optimizer=AdamW lr0=0.001 lrf=0.01 momentum=0.937 weight_decay=0.0005 `
    warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 `
    mosaic=1.0 mixup=0.1 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 `
    close_mosaic=10 patience=20 `
    pretrained=True cls_remap=True `
    assignment_stats=True assignment_small_area=1024.0 assignment_medium_area=9216.0 `
    save=True save_period=1 save_json=True val=True plots=True `
    project="D:/coding/YOLO-Master/A2/runs" name="p2_y26n-p2_vd640_s42_20e" exist_ok=False

Write-Host "Training process exited with code $LASTEXITCODE" -ForegroundColor Yellow