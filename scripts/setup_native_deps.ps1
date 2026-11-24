param(
    [string]$OrtSourceDir = "d:\bgiprro\bgi_publish_new\runtimes\win-x64\native",
    [switch]$DownloadOpenCv
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VendorDir = Join-Path $RepoRoot "vendor"
$OrtVendorDir = Join-Path $VendorDir "ort"
$OpenCvVendorDir = Join-Path $VendorDir "opencv"

New-Item -ItemType Directory -Force -Path $OrtVendorDir | Out-Null

# 1. Copy ONNX Runtime dynamic library for ort (`load-dynamic`)
$ortDll = Join-Path $OrtSourceDir "onnxruntime.dll"
if (Test-Path $ortDll) {
    Copy-Item $ortDll (Join-Path $OrtVendorDir "onnxruntime.dll") -Force
    Write-Host "[OK] Copied onnxruntime.dll to $OrtVendorDir"
} else {
    Write-Warning "onnxruntime.dll not found at $ortDll"
}

foreach ($extra in @("DirectML.dll")) {
    $p = Join-Path $OrtSourceDir $extra
    if (Test-Path $p) {
        Copy-Item $p (Join-Path $OrtVendorDir $extra) -Force
    }
}

# Also place copies beside debug/release targets if target dir exists
foreach ($profile in @("debug", "release")) {
    $targetOut = Join-Path $RepoRoot "target\$profile"
    if (Test-Path $targetOut) {
        Get-ChildItem $OrtVendorDir -Filter "*.dll" | ForEach-Object {
            Copy-Item $_.FullName (Join-Path $targetOut $_.Name) -Force
        }
        $cvBin = Join-Path $OpenCvVendorDir "build\x64\vc16\bin\opencv_world4110.dll"
        if (Test-Path $cvBin) {
            Copy-Item $cvBin (Join-Path $targetOut "opencv_world4110.dll") -Force
        }
    }
}

# 2. Optional OpenCV 4.11.0 official prebuilt SDK setup for Phase 2 (`opencv411` feature)
if ($DownloadOpenCv) {
    $cvArchive = Join-Path $VendorDir "opencv-4.11.0-windows.exe"
    $cvUrl = "https://github.com/opencv/opencv/releases/download/4.11.0/opencv-4.11.0-windows.exe"
    if (!(Test-Path $cvArchive)) {
        Write-Host "[INFO] Downloading OpenCV 4.11.0 SDK from $cvUrl ..."
        Invoke-WebRequest -Uri $cvUrl -OutFile $cvArchive
    }
    if (!(Test-Path (Join-Path $OpenCvVendorDir "build\x64\vc16\lib\opencv_world4110.lib"))) {
        Write-Host "[INFO] Extracting OpenCV 4.11.0 SDK to $VendorDir ..."
        & $cvArchive -o"$VendorDir" -y | Out-Null
    }
}

# 3. Generate .cargo/config.toml if libclang or opencv is present
$cargoDir = Join-Path $RepoRoot ".cargo"
New-Item -ItemType Directory -Force -Path $cargoDir | Out-Null
$cargoCfg = Join-Path $cargoDir "config.toml"

$lines = @("[env]")
$libclang = "C:\Program Files\LLVM\bin"
if (Test-Path (Join-Path $libclang "libclang.dll")) {
    $lines += "LIBCLANG_PATH = `"$($libclang.Replace('\', '\\'))`""
}

$cvInclude = Join-Path $OpenCvVendorDir "build\include"
$cvLib = Join-Path $OpenCvVendorDir "build\x64\vc16\lib"
$ucrtDir = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\Include" -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "ucrt\stdio.h") } |
    Sort-Object Name -Descending |
    Select-Object -First 1 -ExpandProperty FullName

if ((Test-Path $cvInclude) -and (Test-Path (Join-Path $cvLib "opencv_world4110.lib"))) {
    $incList = @($cvInclude)
    if ($ucrtDir) {
        $incList += (Join-Path $ucrtDir "ucrt")
        $incList += (Join-Path $ucrtDir "shared")
        $incList += (Join-Path $ucrtDir "um")
    }
    $incJoined = ($incList -join ",").Replace('\', '\\')
    $lines += "OPENCV_INCLUDE_PATHS = `"$incJoined`""
    $lines += "OPENCV_LINK_PATHS = `"$($cvLib.Replace('\', '\\'))`""
    $lines += 'OPENCV_LINK_LIBS = "opencv_world4110"'
    Write-Host "[OK] OpenCV 4.11 SDK detected and written to .cargo/config.toml"
} else {
    Write-Host "[INFO] OpenCV 4.11 SDK not installed yet. Run with -DownloadOpenCv when ready for Phase 2."
}

$lines -join "`r`n" | Set-Content -Path $cargoCfg -Encoding UTF8
Write-Host "[OK] Wrote $cargoCfg"
