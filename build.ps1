# build.ps1 -- compila o TootegaEngine no Windows com o MSVC do Visual Studio.
#
#   .\build.ps1                compila
#   .\build.ps1 test           compila e roda o selftest
#   .\build.ps1 clean          limpa
#   .\build.ps1 -Arch AVX512   compila com AVX-512 (default: AVX2 = x86-64-v3)
#   .\build.ps1 -Debug         sem otimizacao, com assertions e /Zi
#
# O compilador e localizado por `vswhere`, nunca por caminho fixo.  No Linux o
# build e o `Makefile` (gcc/clang); as fontes sao C11 ISO e servem aos dois.

param(
    [ValidateSet('all', 'test', 'clean')]
    [string]$Target = 'all',
    [ValidateSet('AVX2', 'AVX512', 'SSE2')]
    [string]$Arch = 'AVX2',
    [switch]$Debug
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$BuildDir = Join-Path $PSScriptRoot 'build'
$Exe      = Join-Path $PSScriptRoot 'tootega.exe'

if ($Target -eq 'clean') {
    if (Test-Path $BuildDir) { Remove-Item $BuildDir -Recurse -Force }
    if (Test-Path $Exe)      { Remove-Item $Exe -Force }
    Write-Host 'limpo.'
    exit 0
}

# ---------------------------------------------------------------- toolchain

function Import-VcVars {
    # vcvars64.bat exporta INCLUDE/LIB/PATH.  Rodamos uma vez e importamos o
    # ambiente resultante para este processo; chamadas seguintes reaproveitam.
    if ($env:__TT_VCVARS -eq '1') { return }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw 'vswhere nao encontrado -- Visual Studio nao esta instalado.'
    }

    $vsPath = & $vswhere -latest -products * `
                 -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                 -property installationPath
    if (-not $vsPath) { throw 'nenhum VS com toolset C/C++ x64 instalado.' }

    $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat nao encontrado em '$vsPath'." }

    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue
        }
    }
    $env:__TT_VCVARS = '1'
}

Import-VcVars

# ------------------------------------------------------------------- flags

# /fp:precise: sem contracao a*b+c -> fma e sem reassociacao de somas.  A
# contracao arredonda uma vez em vez de duas e muda bits; a reassociacao muda a
# ordem da reducao.  O criterio de saida do M1 e 32/32 tokens identicos ao
# oraculo (docs/15, D4), entao a semantica do C escrito tem de ser a que se le.
# Verificado no disassembly deste MSVC: `a*b+c` sai como vmulss+vaddss e o laco
# de soma sai serial.  Onde quisermos FMA, usamos o intrinseco explicitamente.
$cflags = @(
    '/nologo', '/c', '/std:c11', '/W4', '/WX-', '/utf-8',
    '/fp:precise', "/arch:$Arch",
    '/GS', '/guard:cf', '/Zc:inline', '/Zc:preprocessor',
    '/D_CRT_SECURE_NO_WARNINGS', '/DWIN32_LEAN_AND_MEAN',
    "/Fo:$BuildDir\"
)

if ($Debug) {
    # /MTd: CRT estatico de debug.  O binario continua sem DLL de runtime.
    $cflags += @('/Od', '/MTd', '/Zi', '/DTT_DEBUG=1', "/Fd:$BuildDir\tootega.pdb")
} else {
    $cflags += @('/O2', '/Oi', '/MT', '/DNDEBUG')
}

# Avisos que nao valem o ruido no MSVC:
#   4100 parametro nao usado -- assinaturas uniformes de kernel
#   4127 expressao condicional constante -- idiomatico em macro
$cflags += @('/wd4100', '/wd4127')

$libs = @('psapi.lib', 'kernel32.lib', 'advapi32.lib', 'ws2_32.lib', 'Synchronization.lib')

# ----------------------------------------------------------------- compila

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$sources = Get-ChildItem (Join-Path $PSScriptRoot 'src\*.c') | Sort-Object Name
if (-not $sources) { throw 'nenhuma fonte em src\.' }

Write-Host ("compilando {0} fontes ({1}{2}) ..." -f $sources.Count, $Arch, $(if ($Debug) { ', debug' } else { '' }))

& cl.exe @cflags @($sources.FullName)
if ($LASTEXITCODE -ne 0) { throw "compilacao falhou ($LASTEXITCODE)" }

$objs = Get-ChildItem (Join-Path $BuildDir '*.obj') | ForEach-Object { $_.FullName }

$ldflags = @('/nologo', "/OUT:$Exe", '/INCREMENTAL:NO')
if ($Debug) { $ldflags += '/DEBUG' } else { $ldflags += '/OPT:REF', '/OPT:ICF' }

& link.exe @ldflags @objs @libs
if ($LASTEXITCODE -ne 0) { throw "linkagem falhou ($LASTEXITCODE)" }

$kb = [math]::Round((Get-Item $Exe).Length / 1KB, 1)
Write-Host "  -> tootega.exe ($kb KB)"

# As ferramentas de Tests/ e tools/ linkam src/ estaticamente.  Compilar so o
# engine as deixa medindo o codigo velho -- e elas nao avisam.  Ja custou horas
# de caca a um bug ja corrigido (docs/17).
#
# `$bin` e nao `$exe`: PowerShell nao distingue maiusculas em nome de variavel,
# entao a variavel do laco SOBRESCREVIA `$Exe` -- e o `selftest` la embaixo
# passava a rodar a ultima ferramenta de `tools/`, nao o engine.
$stale = @()
$built = (Get-Item $Exe).LastWriteTime
foreach ($dir in @('Tests', 'tools')) {
    $d = Join-Path $PSScriptRoot $dir
    if (-not (Test-Path $d)) { continue }
    foreach ($bin in Get-ChildItem (Join-Path $d '*.exe') -ErrorAction SilentlyContinue) {
        if ($bin.LastWriteTime -lt $built) {
            $stale += "$dir\$($bin.Name)"
        }
    }
}
if ($stale) {
    Write-Host ""
    Write-Host "ATENCAO: estes binarios ficaram com o src/ ANTIGO dentro:" -ForegroundColor Red
    foreach ($s in $stale) { Write-Host "  $s" -ForegroundColor Red }
    Write-Host "Rode .\build-all.ps1 -- medir com eles da numero errado sem avisar." -ForegroundColor Red
}

if ($Target -eq 'test') {
    & $Exe selftest
    exit $LASTEXITCODE
}
