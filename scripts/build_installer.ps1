[CmdletBinding()]
param(
    [ValidatePattern('^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$')]
    [string] $Version = "0.3.0-beta",

    # 建置用的 Python（乾淨的環境：requirements-core.txt + requirements-build.txt，見 PACKAGING.md）
    [string] $BuildPython = ".venv-build\Scripts\python.exe",
    [string] $OutputDirectory = "release",
    # 測試用：把模型下載網址的主機換成本機伺服器（例如 http://127.0.0.1:8765），不用真的下載好幾 GB
    [string] $ModelHost = "",
    # 安裝程式大小上限（位元組，不含安裝時才下載的模型）
    [long] $MaximumInstallerBytes = 500000000,
    [switch] $SkipAppBuild,
    [switch] $SkipChecksum,
    # 連到 Hugging Face 核對模型清單的 SHA-256 與大小（要網路；公司代理請先設 HTTPS_PROXY）
    [switch] $CheckModelsOnline,
    [switch] $SignWithReleaseCertificate
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-IsccPath {
    # [Files] 的 download 旗標與 Hash 是 Inno Setup 7 才有的功能
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 7\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 7\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 7\ISCC.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "找不到 Inno Setup 7（ISCC.exe）。請從 https://jrsoftware.org/isdl.php 安裝 Inno Setup 7，再建置安裝程式。"
}

$root = Split-Path -Parent $PSScriptRoot
$python = if ([IO.Path]::IsPathRooted($BuildPython)) { $BuildPython } else { Join-Path $root $BuildPython }
$specPath = Join-Path $root "ScreenTranslator.spec"
$installerScript = Join-Path $root "installer\ScreenTranslator.iss"
$distDirectory = Join-Path $root "dist"
$applicationDirectory = Join-Path $distDirectory "ScreenTranslator"
$applicationPath = Join-Path $applicationDirectory "ScreenTranslator.exe"
$generatedDirectory = Join-Path $root "installer\generated"
$outputPath = if ([IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $root $OutputDirectory }

if (-not (Test-Path -LiteralPath $python)) {
    throw "找不到建置用的 Python：$python。先建立乾淨的環境：python -m venv .venv-build，再 .venv-build\Scripts\python.exe -m pip install -r requirements-core.txt -r requirements-build.txt"
}

Push-Location $root
$previousBuildVersion = $env:SCREEN_TRANSLATOR_BUILD_VERSION
try {
    # 1. 模型清單（安裝時要下載的模型）與程式內的清單一致
    $checkArguments = @("scripts\installer_models.py", "--check")
    if ($CheckModelsOnline) { $checkArguments += "--online" }
    & $python @checkArguments
    if ($LASTEXITCODE -ne 0) { throw "模型清單檢查沒通過。" }

    # 2. 程式本體（PyInstaller，資料夾式）
    if (-not $SkipAppBuild) {
        # GPU 版（Vulkan）的 llama.cpp 函式庫：下載（核對 SHA-256）並確認建置環境的 llama-cpp-python 版本與它相同
        & $python scripts\fetch_vulkan_library.py --check-installed
        if ($LASTEXITCODE -ne 0) { throw "準備 GPU 版 llama.cpp 失敗。" }
        $env:SCREEN_TRANSLATOR_BUILD_VERSION = $Version
        & $python -m PyInstaller --noconfirm --clean --distpath $distDirectory --workpath (Join-Path $root "build") $specPath
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller 建置失敗。" }
    }
    if (-not (Test-Path -LiteralPath $applicationPath -PathType Leaf)) { throw "找不到主程式：$applicationPath" }
    # 兩套 llama.cpp：預設的（lib）一定是 CPU 版，GPU 版（lib_vulkan）要在
    $llamaDirectory = Get-ChildItem -LiteralPath $applicationDirectory -Recurse -Directory -Filter "llama_cpp" | Select-Object -First 1
    if ($null -eq $llamaDirectory) { throw "程式資料夾裡找不到 llama_cpp。" }
    if (Test-Path -LiteralPath (Join-Path $llamaDirectory.FullName "lib\ggml-vulkan.dll")) { throw "預設的 llama.cpp（lib）不能是 GPU 版：沒有 Vulkan 的電腦會整個載入失敗。" }
    if (-not (Test-Path -LiteralPath (Join-Path $llamaDirectory.FullName "lib_vulkan\ggml-vulkan.dll"))) { throw "程式裡沒有 GPU 版 llama.cpp（lib_vulkan）。" }
    # 預設的（lib）CPU 版要用 GPU 版 wheel 的 ggml-cpu.dll（只用 AVX2）：官方 CPU wheel 的含 AVX-512，沒有 AVX-512 的 CPU 會「非法指令」當掉
    $cpuKernel = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $llamaDirectory.FullName "lib\ggml-cpu.dll")).Hash
    $gpuKernel = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $llamaDirectory.FullName "lib_vulkan\ggml-cpu.dll")).Hash
    if ($cpuKernel -ne $gpuKernel) { throw "預設的 llama.cpp（lib）用的不是 AVX2 版的 ggml-cpu.dll（見 scripts/fetch_vulkan_library.py）。" }
    # 最上層不能有 llama.cpp 的 DLL：Windows 會先載入它們，蓋掉 llama_backend.select_library 選好的那一套
    $stray = Get-ChildItem -LiteralPath $applicationDirectory, (Join-Path $applicationDirectory "_internal") -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(ggml.*|llama|mtmd)\.dll$' }
    if ($stray) { throw ("最上層有 llama.cpp 的 DLL（會蓋掉選好的那一套）：" + (($stray | ForEach-Object { $_.FullName }) -join ", ")) }

    # 3. 第三方套件的授權檔（隨安裝程式散布）
    & $python scripts\collect_licenses.py $applicationDirectory
    if ($LASTEXITCODE -ne 0) { throw "收集授權檔失敗。" }

    # 4. 產生安裝程式的模型元件與 [Files]（models.inc 與各模型的 manifest.json）
    $generateArguments = @("scripts\installer_models.py", "--generate", $generatedDirectory)
    if ($ModelHost) { $generateArguments += @("--host", $ModelHost) }
    & $python @generateArguments
    if ($LASTEXITCODE -ne 0) { throw "產生模型元件失敗。" }

    # 5. 安裝程式（Inno Setup 7）
    New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
    $iscc = Get-IsccPath
    $compilerArguments = @(
        "/DMyAppVersion=$Version",
        "/DSourceDir=$applicationDirectory",
        "/DOutputDir=$outputPath"
    )
    if ($SignWithReleaseCertificate) {
        $signScript = Join-Path $root "scripts\sign_release.ps1"
        $signCommand = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$signScript`" -Path `$f"
        $compilerArguments += "/DEnableSigning=1"
        $compilerArguments += "/SScreenTranslator=$signCommand"
    }
    & $iscc @compilerArguments $installerScript
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup 建置安裝程式失敗。" }

    $installerPath = Join-Path $outputPath "ScreenTranslator-$Version-Setup.exe"
    if (-not (Test-Path -LiteralPath $installerPath -PathType Leaf)) { throw "Inno Setup 沒有產生預期的安裝程式：$installerPath" }
    $installerSize = (Get-Item -LiteralPath $installerPath).Length
    if ($installerSize -ge $MaximumInstallerBytes) {
        throw "安裝程式大小為 $installerSize 位元組，超過上限 $MaximumInstallerBytes。"
    }
    Write-Host ("安裝程式：{0}（{1:N1} MB，不含安裝時下載的模型）" -f $installerPath, ($installerSize / 1MB))

    if (-not $SkipChecksum) {
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installerPath).Hash.ToLowerInvariant()
        "$hash *$([IO.Path]::GetFileName($installerPath))" | Set-Content -LiteralPath "$installerPath.sha256" -Encoding ascii -NoNewline
        Write-Host "SHA-256：$hash"
    }
}
finally {
    $env:SCREEN_TRANSLATOR_BUILD_VERSION = $previousBuildVersion
    Pop-Location
}
