# build-tool.ps1 -- compila uma ferramenta nativa.
#
# Conforme CLAUDE.md: a fonte mora em `<pasta>/<nome>/` e o executavel sai em
# `<pasta>/<nome>.exe`. Serve as duas arvores -- `Tests/` para medicao e
# `tools/` para producao -- porque quem chama passa a propria pasta.
#
# A ferramenta REUSA `src/` (menos o `main.c` do engine): e o que garante que
# ela mede o engine, e nao uma reimplementacao dele.

param(
    [Parameter(Mandatory = $true)] [string]$Name,
    [Parameter(Mandatory = $true)] [string]$Dir,
    [ValidateSet('all', 'clean')] [string]$Target = 'all',
    [switch]$Dbg
)

$ErrorActionPreference = 'Stop'

# Raiz do projeto: sobe ate achar `src/`. Nao assume profundidade, porque as
# ferramentas vivem em duas arvores diferentes.
$Here = $Dir
$Root = $Dir
while ($Root -and -not (Test-Path (Join-Path $Root 'src'))) {
    $Root = Split-Path $Root -Parent
}
if (-not $Root) { throw "nao achei a raiz do projeto a partir de '$Dir'." }

$Out = Join-Path (Split-Path $Dir -Parent) "$Name.exe"
$Obj = Join-Path $Dir 'obj'

if (-not (Test-Path (Join-Path $Here "$Name.c"))) {
    throw "$Dir\$Name.c nao existe."
}

if ($Target -eq 'clean') {
    if (Test-Path $Obj) { Remove-Item $Obj -Recurse -Force }
    if (Test-Path $Out) { Remove-Item $Out -Force }
    Write-Host "limpo: $Name"
    exit 0
}

if ($env:__TT_VCVARS -ne '1') {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { throw 'vswhere nao encontrado.' }
    $vsPath = & $vswhere -latest -products * `
                 -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                 -property installationPath
    $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue
        }
    }
    $env:__TT_VCVARS = '1'
}

New-Item -ItemType Directory -Force -Path $Obj | Out-Null

# Mesmas flags do engine.  `/fp:precise` sem contracao e a base de tudo o que
# uma medicao feita por estas ferramentas afirma (docs/15).
$inc_src    = Join-Path $Root 'src'
$inc_shared = Join-Path $Root 'tools\shared'

$cflags = @(
    '/nologo', '/c', '/std:c11', '/W4', '/utf-8', '/fp:precise', '/arch:AVX2',
    '/GS', '/Zc:inline', '/Zc:preprocessor',
    '/D_CRT_SECURE_NO_WARNINGS', '/DWIN32_LEAN_AND_MEAN',
    "/I$inc_src", "/I$inc_shared", '/wd4100', '/wd4127', "/Fo:$Obj\"
)
if ($Dbg) {
    $cflags += @('/Od', '/MTd', '/Zi', "/Fd:$Obj\$Name.pdb")
} else {
    $cflags += @('/O2', '/Oi', '/MT', '/DNDEBUG')
}

$sources = @(Get-ChildItem (Join-Path $inc_src '*.c') |
             Where-Object { $_.Name -ne 'main.c' } |
             ForEach-Object { $_.FullName })
$sources += (Get-ChildItem (Join-Path $inc_shared '*.c') |
             ForEach-Object { $_.FullName })

Write-Host ("compilando {0} ({1} fontes) ..." -f $Name, ($sources.Count + 1))
& cl.exe @cflags @sources
if ($LASTEXITCODE -ne 0) { throw "compilacao falhou ($LASTEXITCODE)" }

# A fonte da ferramenta sai com nome de objeto PROPRIO.
#
# Sem isto, uma ferramenta com o mesmo nome de um arquivo de `src/` grava os
# dois no mesmo `obj\<nome>.obj`. Ja aconteceu, e o sintoma nao aponta para a
# causa: o link falha com LNK1169 (simbolo duplicado) ou LNK1561 (sem entry
# point), dependendo de quem sobrescreveu quem.
& cl.exe @cflags "/Fo:$Obj\tool_$Name.obj" (Join-Path $Here "$Name.c")
if ($LASTEXITCODE -ne 0) { throw "compilacao falhou ($LASTEXITCODE)" }

$objs = Get-ChildItem (Join-Path $Obj '*.obj') | ForEach-Object { $_.FullName }
& link.exe /nologo "/OUT:$Out" /INCREMENTAL:NO $objs psapi.lib kernel32.lib advapi32.lib ws2_32.lib Synchronization.lib
if ($LASTEXITCODE -ne 0) { throw "linkagem falhou ($LASTEXITCODE)" }

Write-Host ("  -> {0} ({1:N1} KB)" -f $Out, ((Get-Item $Out).Length / 1KB))
