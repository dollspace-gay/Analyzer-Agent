; Protocol AI - Inno Setup Installer Script
; Creates a professional Windows installer with everything bundled

#define MyAppName "Protocol AI"
#define MyAppVersion "1.0"
#define MyAppPublisher "Protocol AI Team"
#define MyAppExeName "ProtocolAI.exe"

[Setup]
AppId={{PROTOCOL-AI-GOVERNANCE-LAYER}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=installer_output
OutputBaseFilename=ProtocolAI_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; Main executable (built by PyInstaller)
Source: "dist\ProtocolAI.exe"; DestDir: "{app}"; Flags: ignoreversion

; CUDA Runtime DLLs (if using CUDA)
Source: "cuda_runtime\*.dll"; DestDir: "{app}\cuda_runtime"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: DirExists('cuda_runtime')

; Configuration files
Source: "config\default_config.json"; DestDir: "{app}\config"; Flags: ignoreversion

; README and documentation
Source: "QUICK_START.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "MODULE_LIBRARY_COMPLETE.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "INSTALLATION_COMPLETE.md"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
Name: "{app}\models"; Permissions: users-full
Name: "{app}\output"; Permissions: users-full
Name: "{app}\research_storage"; Permissions: users-full

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
; Launch after installation
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
var
  ModelDownloadPage: TInputFileWizardPage;

procedure InitializeWizard;
begin
  { Create custom page for model file selection }
  ModelDownloadPage := CreateInputFilePage(wpSelectDir,
    'Model File Location', 'Select your GGUF model file',
    'Please select the location of your model file (e.g., DeepSeek-R1-*.gguf). If you don''t have one, you can download it later.');
  ModelDownloadPage.Add('Model file location:',
    'GGUF files|*.gguf|All files|*.*',
    '.gguf');
  ModelDownloadPage.Edits[0].Text := ExpandConstant('{userdocs}\Models\*.gguf');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  ModelPath: String;
begin
  Result := True;

  if CurPageID = ModelDownloadPage.ID then
  begin
    ModelPath := ModelDownloadPage.Values[0];
    if (ModelPath <> '') and FileExists(ModelPath) then
    begin
      { Copy model to installation directory }
      FileCopy(ModelPath, ExpandConstant('{app}\models\model.gguf'), False);
    end;
  end;
end;
