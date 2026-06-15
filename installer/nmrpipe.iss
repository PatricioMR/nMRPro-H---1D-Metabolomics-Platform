; Inno Setup script for nMRPipe
; Build:  iscc installer\nmrpipe.iss   (after PyInstaller produced dist\nMRPipe)
; Output: Output\nMRPipe-Setup.exe
;
; The app seeds its editable databases to %APPDATA%\nMRPipe on first run, so the
; install folder (Program Files) can stay read-only.

#define AppName    "nMRPipe"
#define AppVersion "0.2"
#define AppExe     "nMRPipe.exe"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=nMRPipe
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\Output
OutputBaseFilename=nMRPipe-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\utils\nmrpipe.ico
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; Everything PyInstaller produced in dist\nMRPipe
Source: "..\dist\nMRPipe\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}";        Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";  Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
