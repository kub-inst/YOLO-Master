param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(0.0, 1.0)]
    [double]$Lambda
)

$ErrorActionPreference = "Stop"
Set-Location "D:\coding\YOLO-Master"

$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$lambdaText = $Lambda.ToString("0.00", $invariant)
$runName = "lambda_$($lambdaText.Replace('.', 'p'))"
$runRoot = "D:/coding/YOLO-Master/A2/runs/p1_dynamic_topk_lambda_sweep_b2_w0"
$runDir = "$runRoot/$runName"

Write-Host "Dynamic TopK lambda=$lambdaText | batch=2 workers=0 | 10 epochs"
Write-Host "Output: $runDir"

conda run --no-capture-output -n yolo_master yolo detect train `
    model="A2/configs/yolo26n_visdrone.yaml" `
    data="A2/configs/visdrone.yaml" `
    epochs=10 `
    imgsz=640 `
    batch=2 `
    workers=0 `
    device=0 `
    optimizer=AdamW `
    seed=42 `
    deterministic=True `
    pretrained=False `
    cache=False `
    rect=False `
    cos_lr=False `
    close_mosaic=10 `
    tal_dynamic_topk_small=True `
    tal_dynamic_topk_lambda=$lambdaText `
    project=$runRoot `
    name=$runName `
    exist_ok=False

if ($LASTEXITCODE -ne 0) {
    throw "Training failed (exit code $LASTEXITCODE)."
}

conda run --no-capture-output -n yolo_master python A2/scripts/evaluate_p0_checkpoints.py `
    --weights="$runDir/weights" `
    --data="A2/configs/visdrone.yaml" `
    --images="D:/datasets/VisDrone2019-DET-val/images" `
    --labels="D:/datasets/VisDrone2019-DET-val/labels" `
    --output="$runDir/checkpoint_area_metrics.json" `
    --imgsz=640 `
    --batch=2 `
    --device=0 `
    --workers=0 `
    --start-epoch=1 `
    --end-epoch=10

if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint evaluation failed (exit code $LASTEXITCODE)."
}

Write-Host "Completed lambda=$lambdaText."
