$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\coding\YOLO-Master"

# Throughput-oriented variant. It keeps the Dynamic TopK ablation and all optimization/augmentation values unchanged,
# while raising the batch size to use the otherwise idle RTX 5060 Laptop GPU.
# This run is not directly interchangeable with the batch=2 P1 result: a larger per-step mini-batch changes gradient
# noise even though Ultralytics preserves the nominal batch size (nbs=64) through gradient accumulation.
Write-Host "Starting P1 fast: YOLO26n + VisDrone small-GT Dynamic TopK, 10 epochs" -ForegroundColor Cyan
Write-Host "Throughput settings: batch=16, workers=0; if CUDA OOM occurs, rerun with batch=8." -ForegroundColor Cyan
Write-Host "Small GTs: K=ceil(0.8*x); medium/large GTs: fixed baseline K=10" -ForegroundColor Cyan

conda run --no-capture-output -n yolo_master yolo detect train `
    model="D:/coding/YOLO-Master/yolo26n.pt" `
    data="D:/coding/YOLO-Master/A2/configs/visdrone.yaml" `
    epochs=10 imgsz=640 batch=16 device=0 workers=0 `
    seed=42 deterministic=True `
    optimizer=AdamW lr0=0.001 lrf=0.01 momentum=0.937 weight_decay=0.0005 `
    warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 `
    mosaic=1.0 mixup=0.1 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 `
    close_mosaic=10 patience=50 `
    assignment_stats=True assignment_small_area=1024.0 assignment_medium_area=9216.0 `
    tal_dynamic_topk_small=True tal_dynamic_topk_lambda=0.8 `
    save=True save_period=1 save_json=True val=True plots=True `
    project="D:/coding/YOLO-Master/A2/runs" name="p1_dynamic_topk_a08_y26n_vd640_s42_10e_fast_b16_w0" exist_ok=False

Write-Host "Training process exited with code $LASTEXITCODE" -ForegroundColor Yellow
