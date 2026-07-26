# build-all.ps1 -- o unico ponto de entrada de build.
#
#   .\build-all.ps1              compila o que estiver desatualizado
#   .\build-all.ps1 test         idem + selftest do engine
#   .\build-all.ps1 -Force       recompila tudo
#   .\build-all.ps1 clean        limpa tudo
#   .\build-all.ps1 -Only bench  so um alvo (ou 'engine')
#
# ---------------------------------------------------------------------------
# Por que este script existe, e por que ele carimba os binarios
#
# As ferramentas de `Tests/` e `tools/` linkam `src/` ESTATICAMENTE. Depois de
# corrigir um bug no forward, rodar so `.\build.ps1` deixa cada ferramenta com
# o codigo velho dentro -- e elas continuam medindo, sem avisar, a versao
# errada. Isso aconteceu: horas perseguindo um bug ja corrigido, porque o
# `ablate.exe` era anterior ao conserto.
#
# Duas defesas, e nenhuma depende de alguem lembrar:
#
#   1. Obsolescencia detectada. Qualquer fonte de `src/` mais nova que um
#      executavel obriga a reconstrucao dele.
#   2. Carimbo. `src/build_id.h` recebe o hash das fontes, e todo binario o
#      imprime. Numero medido por binario velho passa a se denunciar.
# ---------------------------------------------------------------------------

param(
    [ValidateSet('all', 'test', 'clean')] [string]$Target = 'all',
    [string]$Only = '',
    [switch]$Force,
    [switch]$Dbg
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$SrcDir = Join-Path $Root 'src'

# --------------------------------------------------------------- carimbo

function Get-SourceFiles {
    # `build_id.h` fica de fora: ele e DERIVADO do hash, e incluir-se na
    # propria conta faria o id nunca estabilizar.
    Get-ChildItem (Join-Path $SrcDir '*.c'), (Join-Path $SrcDir '*.h') |
        Where-Object { $_.Name -ne 'build_id.h' } |
        Sort-Object Name
}

function Get-BuildId {
    $sb = [System.Text.StringBuilder]::new()
    foreach ($f in Get-SourceFiles) {
        [void]$sb.Append((Get-FileHash $f.FullName -Algorithm SHA256).Hash)
    }
    $bytes = [System.Text.Encoding]::ASCII.GetBytes($sb.ToString())
    $sha = [System.Security.Cryptography.SHA256]::Create()
    ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString('x2') }) -join '' `
        | ForEach-Object { $_.Substring(0, 12) }
}

function Write-BuildIdHeader([string]$id) {
    $path = Join-Path $SrcDir 'build_id.h'
    $want = @(
        '/* build_id.h -- GERADO por build-all.ps1.  Nao edite, nao versione.',
        ' *',
        ' * Hash das fontes de src/.  Todo binario o imprime, para que um',
        ' * executavel desatualizado se denuncie ao medir (docs/10, docs/17).',
        ' */',
        "#define TT_BUILD_ID `"$id`"",
        ''
    ) -join "`r`n"

    # So reescreve se mudou: tocar o arquivo a toa invalidaria o cache de
    # obsolescencia que este mesmo script usa.
    if (-not (Test-Path $path) -or (Get-Content $path -Raw) -ne $want) {
        Set-Content $path -Value $want -Encoding utf8 -NoNewline
    }
}

# ----------------------------------------------------------------- alvos

function Find-Builds([string]$dir) {
    if (-not (Test-Path $dir)) { return @() }
    Get-ChildItem $dir -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName 'build.ps1') } |
        ForEach-Object {
            [pscustomobject]@{
                Name   = $_.Name
                Dir    = $_.FullName
                Script = Join-Path $_.FullName 'build.ps1'
                Exe    = Join-Path $dir "$($_.Name).exe"
            }
        }
}

$targets = @(
    [pscustomobject]@{ Name = 'engine'; Dir = $Root
                       Script = Join-Path $Root 'build.ps1'
                       Exe = Join-Path $Root 'tootega.exe' }
)
$targets += Find-Builds (Join-Path $Root 'tools')
$targets += Find-Builds (Join-Path $Root 'Tests')

if ($Only) {
    $targets = @($targets | Where-Object { $_.Name -eq $Only })
    if (-not $targets) { throw "nao achei o alvo '$Only'." }
}

if ($Target -eq 'clean') {
    foreach ($t in $targets) { & $t.Script clean }
    $bid = Join-Path $SrcDir 'build_id.h'
    if (Test-Path $bid) { Remove-Item $bid -Force }
    Write-Host 'limpo.'
    exit 0
}

# ------------------------------------------------------- obsolescencia

$buildId = Get-BuildId
Write-BuildIdHeader $buildId
Write-Host "build id $buildId" -ForegroundColor DarkGray

# A fonte mais nova de src/ (mais o build_id.h recem-escrito) define o piso:
# executavel anterior a ela esta medindo codigo velho.
$srcNewest = (Get-SourceFiles | Sort-Object LastWriteTime -Descending |
              Select-Object -First 1).LastWriteTime
$bidFile = Join-Path $SrcDir 'build_id.h'
if (Test-Path $bidFile) {
    $t = (Get-Item $bidFile).LastWriteTime
    if ($t -gt $srcNewest) { $srcNewest = $t }
}

function Test-Stale($t) {
    if ($Force) { return $true }
    if (-not (Test-Path $t.Exe)) { return $true }

    $exeTime = (Get-Item $t.Exe).LastWriteTime
    if ($exeTime -lt $srcNewest) { return $true }

    # Fontes proprias da ferramenta e o seu build.ps1.
    $own = Get-ChildItem $t.Dir -File -Filter '*.c' -ErrorAction SilentlyContinue
    foreach ($f in @($own) + @(Get-Item $t.Script)) {
        if ($f.LastWriteTime -gt $exeTime) { return $true }
    }
    return $false
}

# ------------------------------------------------------- sintaxe da UI

# A UI web e um literal de string C. Um `\n` a mais no lugar errado vira
# backslash-n DENTRO do codigo JS, o script inteiro deixa de ser parseado e a
# pagina serve 200 com todos os listeners mortos -- foi o que aconteceu com a
# tela inicial. O compilador nao ve, o teste de HTTP tampouco: os bytes chegam.
#
# Entao o unico jeito de pegar isto no build e parsear o JS. Se `node` nao
# existir, o build segue -- e checagem extra, nao dependencia.
function Test-WebUi([string]$root) {
    $node = Get-Command node -ErrorAction SilentlyContinue
    if (-not $node) { return $true }

    $src = Get-Content (Join-Path $root 'src\webui.h') -Raw
    $src = $src.Substring($src.IndexOf('TT_WEBUI_HTML'))

    $sb = [System.Text.StringBuilder]::new()
    foreach ($m in [regex]::Matches($src, '"((?:[^"\\]|\\.)*)"')) {
        [void]$sb.Append([regex]::Replace($m.Groups[1].Value, '\\(.)', {
            param($e)
            switch ($e.Groups[1].Value) {
                'n'  { "`n" } 't' { "`t" } 'r' { "`r" }
                '"'  { '"' }  "'" { "'" }  '\' { '\' }
                default { '\' + $e.Groups[1].Value }
            }
        }))
    }
    $html = $sb.ToString()

    $i = $html.IndexOf('<script>') + 8
    $j = $html.IndexOf('</script>')
    if ($i -lt 8 -or $j -lt $i) {
        Write-Host '  webui: nao achei o <script>' -ForegroundColor Red
        return $false
    }

    $tmp = Join-Path $env:TEMP 'tt_webui_check.js'
    Set-Content $tmp -Value $html.Substring($i, $j - $i) -Encoding utf8 -NoNewline
    $out = & node --check $tmp 2>&1
    Remove-Item $tmp -ErrorAction SilentlyContinue
    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host '=== webui ' .PadRight(64, '=') -ForegroundColor Red
        Write-Host '  JS embutido nao compila -- a pagina abriria inerte:' -ForegroundColor Red
        $out | Select-Object -First 6 | ForEach-Object { Write-Host "  $_" }
        return $false
    }
    return $true
}

# ---------------------------------------------------------------- build

if (-not (Test-WebUi $Root)) { $script:webuiBad = $true }

$forward = @()
if ($Dbg) { $forward += '-Dbg' }

$report = @()
$failed = 0
$t0 = Get-Date

foreach ($t in $targets) {
    if (-not (Test-Stale $t)) {
        $report += [pscustomobject]@{
            Alvo = $t.Name; Estado = 'ja atual'
            KB = [math]::Round((Get-Item $t.Exe).Length / 1KB, 1); Segundos = 0
        }
        continue
    }

    Write-Host ""
    Write-Host ("=== {0} " -f $t.Name).PadRight(64, '=') -ForegroundColor Cyan

    $start = Get-Date
    $ok = $true
    try {
        & $t.Script @forward
        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) { $ok = $false }
    } catch {
        # Uma ferramenta quebrada nao pode esconder as outras.
        Write-Host "  FALHOU: $_" -ForegroundColor Red
        $ok = $false
    }
    if (-not $ok) { $failed++ }

    $report += [pscustomobject]@{
        Alvo     = $t.Name
        Estado   = if ($ok) { 'compilado' } else { 'FALHOU' }
        KB       = if ($ok -and (Test-Path $t.Exe)) {
                       [math]::Round((Get-Item $t.Exe).Length / 1KB, 1) } else { $null }
        Segundos = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)
    }
}

Write-Host ""
Write-Host ("=" * 64)
$report | Format-Table -AutoSize

# Ultima checagem: nenhum executavel pode ter sobrado para tras. E o que
# transforma "eu esqueci" em erro visivel em vez de medida errada.
$stale = @($targets | Where-Object { (Test-Path $_.Exe) -and
                                     (Get-Item $_.Exe).LastWriteTime -lt $srcNewest })
if ($stale) {
    Write-Host ("ATENCAO: {0} desatualizado(s): {1}" -f $stale.Count,
                (($stale | ForEach-Object { $_.Name }) -join ', ')) -ForegroundColor Red
    exit 1
}

$total = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
if ($script:webuiBad) {
    Write-Host "UI web com erro de sintaxe -- veja acima ($total s)" -ForegroundColor Red
    exit 1
}
if ($failed) {
    Write-Host "$failed alvo(s) falharam em $total s" -ForegroundColor Red
    exit 1
}
Write-Host "tudo em dia ($buildId) em $total s"

if ($Target -eq 'test') {
    Write-Host ""
    Write-Host ("=== selftest ").PadRight(64, '=') -ForegroundColor Cyan
    & (Join-Path $Root 'tootega.exe') selftest
    exit $LASTEXITCODE
}
