$ErrorActionPreference = "Stop"
Set-Location "D:\coding\YOLO-Master"

$runRoot = "D:/coding/YOLO-Master/A2/runs"
$runName = "p0_master_v01n_vd800_s42_120e"

Write-Host "Starting baseline: YOLO-Master v0.1-n | VisDrone DET" -ForegroundColor Cyan
Write-Host "imgsz=800 | epochs=120 | patience=0 | full val=548 images" -ForegroundColor Cyan
Write-Host "Output: $runRoot/$runName" -ForegroundColor Cyan

conda run --no-capture-output -n yolo_master yolo detect train `
    model="D:/coding/YOLO-Master/ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml" `
    data="D:/coding/YOLO-Master/A2/configs/visdrone.yaml" `
    epochs=120 imgsz=800 batch=2 device=0 workers=0 `
    seed=42 deterministic=True `
    optimizer=AdamW lr0=0.001 lrf=0.01 momentum=0.937 weight_decay=0.0005 `
    warmup_epochs=3.0 warmup_momentum=0.8 warmup_bias_lr=0.1 `
    mosaic=1.0 mixup=0.1 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 `
    close_mosaic=10 patience=0 `
    assignment_stats=True assignment_small_area=1024.0 assignment_medium_area=9216.0 `
    save=True save_period=1 save_json=True val=True plots=True `
    project=$runRoot name=$runName exist_ok=False

Write-Host "Training process exited with code $LASTEXITCODE" -ForegroundColor Yellow
if ($LASTEXITCODE -ne 0) {
    throw "Baseline training failed."
}
