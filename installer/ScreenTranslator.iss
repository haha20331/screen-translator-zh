; 螢幕截圖翻譯的安裝程式（Inno Setup 7，每位使用者安裝，不需要系統管理員權限）。
;
; 程式本體是 PyInstaller 的資料夾輸出（ScreenTranslator.spec）。翻譯模型（GGUF，每個 1.9～4.7 GB）不在安裝程式裡：
; 依使用者在「選擇元件」頁的勾選，安裝時從 Hugging Face 下載，並用 SHA-256 驗證（清單：models.json，由
; scripts/installer_models.py 產生 generated\models.inc）。預設勾選推薦的 Qwen2.5 3B。
;
; 用 scripts/build_installer.ps1 建置（它會先檢查清單、產生 models.inc 再呼叫 ISCC）。
;
; 命令列（靜默安裝）：  Setup.exe /VERYSILENT /COMPONENTS="app,models,models\qwen25_3b"   只裝程式：/COMPONENTS="app"

#ifndef MyAppVersion
  #define MyAppVersion "0.3.0-beta"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\ScreenTranslator"
#endif
#ifndef OutputDir
  #define OutputDir "..\release"
#endif

#define MyAppName "Screen Translator"
#define MyAppExeName "ScreenTranslator.exe"

[Setup]
AppId={{87C8FDB0-243F-4A8F-A399-549C25402E72}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
DefaultDirName={autopf}\ScreenTranslator
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=ScreenTranslator-{#MyAppVersion}-Setup
SetupIconFile=..\assets\app_launch_v4.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
LicenseFile=..\LICENSE
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
ShowLanguageDialog=no
#ifdef EnableSigning
SignTool=ScreenTranslator
SignedUninstaller=yes
#endif

[Languages]
Name: "zh_tw"; MessagesFile: "compiler:Languages\ChineseTraditional.isl"

[Types]
Name: "recommended"; Description: "建議安裝：程式＋Qwen2.5 3B 翻譯模型（約 2.1 GB，安裝時下載）"
Name: "minimal"; Description: "只安裝程式（稍後自己下載或放入翻譯模型）"
Name: "custom"; Description: "自訂：自己勾選要下載的翻譯模型"; Flags: iscustom

[Components]
Name: "app"; Description: "Screen Translator 程式"; Types: recommended minimal custom; Flags: fixed
; 「翻譯模型」群組與每個模型的元件由 scripts/installer_models.py 產生，見 [Code] 前面的 #include

[Tasks]
Name: "desktopicon"; Description: "建立桌面捷徑"; GroupDescription: "附加工作："; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Components: app; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Components: app; Flags: ignoreversion
Source: "..\THIRD_PARTY_LICENSES.md"; DestDir: "{app}"; Components: app; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Components: app; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "啟動 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 模型是安裝程式下載的，解除安裝會一起移除；使用者自己放進來的模型留著，資料夾清空時才一併刪除
Type: dirifempty; Name: "{app}\models\translation"
Type: dirifempty; Name: "{app}\models"

; 翻譯模型的元件與下載（[Components]、[Files]；每個模型 SHA-256 驗證）。必須放在 [Code] 之前
#include "generated\models.inc"

[Code]
const
  ResearchLicenseComponent = 'models\qwen25_3b';

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = wpSelectComponents then
  begin
    if WizardIsComponentSelected(ResearchLicenseComponent) then
    begin
      // Qwen2.5 3B 的授權（Qwen Research License）不是一般的開源授權，不能隨程式散布，所以改成讓使用者自己下載，並在這裡說清楚
      if SuppressibleMsgBox(
           'Qwen2.5 3B 模型的授權是「Qwen Research License」，只能用於研究與評估，不可用於商業用途。' + #13#10 +
           '模型檔會在安裝時從 Hugging Face 下載，授權全文會放在模型資料夾的 LICENSE 檔。' + #13#10 + #13#10 +
           '要依這個授權下載嗎？', mbConfirmation, MB_YESNO, IDYES) = IDNO then
      begin
        Result := False;
        Exit;
      end;
    end;
    // WizardSelectedComponents(False) 是用逗號隔開的元件名稱，例如 app,models,models\qwen25_3b；沒有任何 models\ 開頭的就是沒選模型
    if Pos('models\', WizardSelectedComponents(False)) = 0 then
    begin
      if SuppressibleMsgBox(
           '沒有選任何翻譯模型，程式第一次翻譯時會找不到模型。' + #13#10 +
           '之後可以重新執行這個安裝程式補下載，或把模型檔放進安裝資料夾的 models\translation 底下。' + #13#10 + #13#10 +
           '仍要只安裝程式嗎？', mbConfirmation, MB_YESNO, IDYES) = IDNO then
        Result := False;
    end;
  end;
end;

