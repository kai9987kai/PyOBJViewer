; Inno Setup script for PyOBJViewer
; Build:  ISCC.exe packaging\installer.iss   (run from the repo root)

#define MyAppName "PyOBJViewer"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "kai9987kai"
#define MyAppURL "https://github.com/kai9987kai/PyOBJViewer"
#define MyAppExeName "PyOBJViewer.exe"

[Setup]
AppId={{8E2F4C6A-1B3D-4E5F-9A7C-D2E8F1A0B345}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
; Per-user install: no admin prompt, lands in %LocalAppData%\Programs.
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=PyOBJViewer-Setup-{#MyAppVersion}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "assocobj"; Description: "Open .obj files with {#MyAppName}"; GroupDescription: "File associations:"; Flags: unchecked
Name: "assocstl"; Description: "Open .stl files with {#MyAppName}"; GroupDescription: "File associations:"; Flags: unchecked

[Files]
Source: "..\dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\examples\torus.obj"; DestDir: "{app}\examples"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{#MyAppName} sample (torus)"; Filename: "{app}\{#MyAppExeName}"; Parameters: """{app}\examples\torus.obj"""
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; ProgId shared by both associations.
Root: HKA; Subkey: "Software\Classes\PyOBJViewer.Model"; ValueType: string; ValueData: "3D model"; Flags: uninsdeletekey; Tasks: assocobj assocstl
Root: HKA; Subkey: "Software\Classes\PyOBJViewer.Model\DefaultIcon"; ValueType: string; ValueData: "{app}\{#MyAppExeName},0"; Tasks: assocobj assocstl
Root: HKA; Subkey: "Software\Classes\PyOBJViewer.Model\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: assocobj assocstl
Root: HKA; Subkey: "Software\Classes\.obj\OpenWithProgids"; ValueType: string; ValueName: "PyOBJViewer.Model"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assocobj
Root: HKA; Subkey: "Software\Classes\.obj"; ValueType: string; ValueData: "PyOBJViewer.Model"; Flags: uninsdeletevalue; Tasks: assocobj
Root: HKA; Subkey: "Software\Classes\.stl\OpenWithProgids"; ValueType: string; ValueName: "PyOBJViewer.Model"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assocstl
Root: HKA; Subkey: "Software\Classes\.stl"; ValueType: string; ValueData: "PyOBJViewer.Model"; Flags: uninsdeletevalue; Tasks: assocstl

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
