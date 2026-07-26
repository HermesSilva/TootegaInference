# build.ps1 -- compila tootega_cuda.dll.
#
# Fica separado do `build-tool.ps1` porque nao e uma ferramenta que linka
# `src/`: e uma DLL de backend, compilada por nvcc, que o engine carrega em
# execucao (docs/10).  O engine nunca linka cudart.
#
# nvcc e localizado por CUDA_PATH ou pelas instalacoes conhecidas -- nunca por
# caminho fixo, pela mesma razao que o MSVC vem do vswhere.

param(
    [ValidateSet('all', 'clean')] [string]$Target = 'all',
    [string]$Arch = 'sm_75'          # RTX 2060 SUPER; Turing
)

$ErrorActionPreference = 'Stop'
$Here = $PSScriptRoot
$Root = Split-Path (Split-Path $Here -Parent) -Parent
$Out  = Join-Path $Root 'tootega_cuda.dll'

if ($Target -eq 'clean') {
    foreach ($f in @($Out, (Join-Path $Root 'tootega_cuda.lib'),
                     (Join-Path $Root 'tootega_cuda.exp'))) {
        if (Test-Path $f) { Remove-Item $f -Force }
    }
    Write-Host 'limpo.'
    exit 0
}

function Find-Nvcc {
    if ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) {
        return "$env:CUDA_PATH\bin\nvcc.exe"
    }
    $cmd = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $base = Join-Path ${env:ProgramFiles} 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $base) {
        $v = Get-ChildItem $base -Directory |
             Sort-Object Name -Descending |
             Where-Object { Test-Path (Join-Path $_.FullName 'bin\nvcc.exe') } |
             Select-Object -First 1
        if ($v) { return (Join-Path $v.FullName 'bin\nvcc.exe') }
    }
    throw 'nvcc nao encontrado. Instale o CUDA Toolkit ou defina CUDA_PATH.'
}

$nvcc = Find-Nvcc
Write-Host "nvcc: $nvcc"
& $nvcc --version | Select-Object -Last 2 | ForEach-Object { Write-Host "  $_" }

# O host compiler do nvcc e o cl.exe, e o CUDA 12.6 so aceita VS 2017..2022 --
# o toolset do VS 2026 e recusado por `host_config.h`.
#
# Isto NAO e problema: a DLL fala C ABI e nada mais. O host compiler dela e
# independente do que compila o engine, e forcar `-allow-unsupported-compiler`
# seria trocar uma toolchain suportada por um risco silencioso de runtime.
#
# Entao o engine usa o VS mais novo e o backend usa o mais novo SUPORTADO.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = & $vswhere -products * -version '[17.0,18.0)' `
             -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
             -property installationPath | Select-Object -First 1

if (-not $vsPath) {
    throw ("nenhum Visual Studio 2022 encontrado. O CUDA $((Split-Path (Split-Path $nvcc -Parent) -Leaf)) " +
           'exige host compiler entre 2017 e 2022; instale as Build Tools do ' +
           'VS 2022 ou um CUDA mais novo.')
}
Write-Host "host compiler: $vsPath"

# Ambiente proprio: nao reaproveita `__TT_VCVARS`, que pode ter sido montado
# com o VS 2026 por outro build desta mesma sessao.
#
# E preciso limpar `VSCMD_VER` antes: o `vcvars64.bat` se RECUSA a reconfigurar
# quando ve essa variavel, e sai calado deixando o ambiente do VS anterior no
# lugar. Compilando sozinho isto nunca aparece; dentro do `build-all.ps1`, o
# engine ja passou por aqui e o nvcc acaba achando o toolset de 2026.
foreach ($v in 'VSCMD_VER', 'VSCMD_ARG_TGT_ARCH', 'VSCMD_ARG_HOST_ARCH',
                'VCToolsVersion', 'VCINSTALLDIR', 'VCToolsInstallDir',
                'INCLUDE', 'LIB', 'LIBPATH', '__VSCMD_PREINIT_PATH',
                'VSINSTALLDIR', 'VisualStudioVersion', 'WindowsSdkDir') {
    Remove-Item "env:$v" -ErrorAction SilentlyContinue
}

$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2] `
                 -ErrorAction SilentlyContinue
    }
}

$src = Join-Path $Here 'tootega_cuda.cu'
$inc = Join-Path $Root 'src'

# `-ccbin` aponta o host compiler DIRETAMENTE, em vez de depender do que o
# ambiente deixou no PATH.  Limpar `VSCMD_VER` nao basta: o `vcvars64.bat`
# ainda deixa o toolset anterior a frente em algumas ordens de chamada, e o
# nvcc acaba achando o cl.exe de 2026 -- que o CUDA 12.6 recusa.
$ccbin = Get-ChildItem (Join-Path $vsPath 'VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe') |
         Sort-Object FullName -Descending | Select-Object -First 1
if (-not $ccbin) { throw "cl.exe do VS 2022 nao encontrado em '$vsPath'." }
Write-Host "ccbin: $($ccbin.Directory)"

Write-Host "compilando tootega_cuda.dll ($Arch) ..."
& $nvcc --shared -O3 "-arch=$Arch" "-I$inc" `
        -ccbin "$($ccbin.Directory)" `
        -Xcompiler "/MD,/O2,/W3" `
        -o $Out $src
if ($LASTEXITCODE -ne 0) { throw "nvcc falhou ($LASTEXITCODE)" }

Write-Host ("  -> {0} ({1:N1} KB)" -f $Out, ((Get-Item $Out).Length / 1KB))
Write-Host ""
Write-Host "A DLL depende de cudart64_*.dll. Se o engine reclamar de"
Write-Host "'dependencia nao achada', o PATH e que falta -- nao o arquivo."
