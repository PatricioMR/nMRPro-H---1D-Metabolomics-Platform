; Inno Setup script for NMRPro-H
; Build:  iscc installer\nmrpro_h.iss   (after PyInstaller produced dist\NMRPro-H)
; Output: Output\NMRPro-H-Setup.exe
;
; The app seeds its editable databases to %APPDATA%\NMRPro-H on first run, so the
; install folder (Program Files) can stay read-only.

#define AppName    "NMRPro-H"
; Version is passed by build_windows.bat (iscc /DAppVersion=...), extracted from
; app_paths.__version__ — the single source of truth. Fallback kept so a manual
; `iscc installer\nmrpro_h.iss` still compiles; keep it matching app_paths.
#ifndef AppVersion
  #define AppVersion "0.73"
#endif
#define AppExe     "NMRPro-H.exe"

[Setup]
; Stable identity, independent of the display name. Without it Inno derives the
; AppId from AppName, so renaming the product turns every future release into a
; separate install: no upgrade, and the previous entry left behind in Add/Remove
; Programs. Never change this GUID.
AppId={{cbda7d6b-6c59-53ea-b354-f28c59c0a6b5}}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=NMRPro-H
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\Output
OutputBaseFilename=NMRPro-H-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\utils\nmrpro_h.ico
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; Everything PyInstaller produced in dist\NMRPro-H
Source: "..\dist\NMRPro-H\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}";        Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";  Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
