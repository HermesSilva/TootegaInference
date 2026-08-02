# download-model.ps1 -- fetch the domain container from the private Hub repo.
#
# 19.32 GB, so this is a foreground job you watch: `hf download` resumes on
# interruption, and running it detached hides the one thing worth seeing, which
# is whether it is moving.
#
# What comes down:
#   glm47flash-v5-sft.tootega   the int4 container the engine reads
#   tokenizer.json              required separately -- `tootega generate` takes
#                               it as its own argument, it is not in the container
#
# Then:  python finetune\ask.py "O que e a Arena?"
#
#   .\finetune\download-model.ps1 -Token hf_xxx
#   $env:HF_TOKEN = 'hf_xxx'; .\finetune\download-model.ps1
#   .\finetune\download-model.ps1 -Token hf_xxx -Verify   # check an existing copy

[CmdletBinding()]
param(
    [string]$Token  = $env:HF_TOKEN,
    [string]$RepoId = 'HermesX21/cosmologia-composicional-lora',
    [string]$Dest   = (Join-Path (Split-Path $PSScriptRoot -Parent) 'models\v5-sft'),
    [switch]$Verify
)

$ErrorActionPreference = 'Stop'

$expected = @{
    'glm47flash-v5-sft.tootega' = 19320347926L
    'tokenizer.json'            = 20217442L
}

function Show-State($dest) {
    $ok = $true
    foreach ($name in ($expected.Keys | Sort-Object)) {
        $p = Join-Path $dest $name
        $len = if (Test-Path $p) { (Get-Item $p).Length } else { 0L }

        # A download in flight lives under .cache/huggingface/download as a
        # .incomplete blob, not at the final path -- without looking there, a
        # resumed run reports "missing" for a file that is 80% down.
        if ($len -eq 0) {
            $partial = Get-ChildItem -Recurse -File `
                       -Path (Join-Path $dest '.cache') -Filter '*.incomplete' `
                       -ErrorAction SilentlyContinue |
                       Sort-Object Length -Descending | Select-Object -First 1
            if ($partial -and $expected[$name] -gt 1GB) { $len = $partial.Length }
        }

        $want = $expected[$name]
        if ($len -eq $want) {
            Write-Host ("  ok       {0,-30} {1,7:N2} GB" -f $name, ($len / 1GB))
        } elseif ($len -gt 0) {
            $ok = $false
            Write-Host ("  partial  {0,-30} {1,7:N2} of {2:N2} GB  ({3:P0})" -f `
                        $name, ($len / 1GB), ($want / 1GB), ($len / $want))
        } else {
            $ok = $false
            Write-Host ("  missing  {0}" -f $name)
        }
    }
    return $ok
}

if ($Verify) {
    Write-Host "verifying $Dest`n"
    if (Show-State $Dest) { Write-Host "`ncomplete." ; exit 0 }
    Write-Host "`nincomplete -- run without -Verify to resume." ; exit 1
}

if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    throw "hf CLI not found. Install with: pip install -U huggingface_hub"
}

# Credentials, in order: an explicit -Token, then whatever `hf auth login` left
# on this machine.  A token is NOT required.
#
# And HF_TOKEN is never set from here.  The environment variable outranks the
# stored login, so exporting a stale or empty one turns a working machine into
# a 401 against a private repo -- which is exactly how this script failed the
# first time. When a token is given it goes on the command line, where it
# applies to this call and nothing else.
$auth = @()
if ($Token) {
    $auth = @('--token', $Token)
    Write-Host 'auth: explicit -Token'
} else {
    $who = (& hf auth whoami 2>&1 | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or $who -notmatch 'user=') {
        throw "not logged in and no -Token given. Run `hf auth login`, or pass -Token."
    }
    Write-Host "auth: stored login ($who)"
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

# Free space is checked against what is still MISSING, not the full size -- a
# resumed download does not need room for the bytes already on disk.
$have = 0L
foreach ($name in $expected.Keys) {
    $p = Join-Path $Dest $name
    if (Test-Path $p) { $have += (Get-Item $p).Length }
}
$need = ($expected.Values | Measure-Object -Sum).Sum - $have
$drive = (Get-Item $Dest).PSDrive
$free = $drive.Free
Write-Host ("destination {0}" -f $Dest)
Write-Host ("free {0:N0} GB on {1}:  need {2:N1} GB more`n" -f `
            ($free / 1GB), $drive.Name, ($need / 1GB))
if ($free -lt $need * 1.05) {
    throw ("not enough space: {0:N1} GB free, {1:N1} GB needed" -f ($free/1GB), ($need/1GB))
}

Write-Host "current state:"
Show-State $Dest | Out-Null

Write-Host "`ndownloading (resumes if interrupted; Ctrl-C is safe) ...`n"
$sw = [Diagnostics.Stopwatch]::StartNew()

& hf download $RepoId 'glm47flash-v5-sft.tootega' 'tokenizer.json' --local-dir $Dest @auth
if ($LASTEXITCODE -ne 0) { throw "download failed ($LASTEXITCODE) -- rerun to resume" }

$sw.Stop()
Write-Host ("`ndone in {0:N1} min" -f $sw.Elapsed.TotalMinutes)
if (Show-State $Dest) {
    Write-Host "`nsizes match. Try it:"
    Write-Host '  python finetune\ask.py "Na morte, a Mente individual dissolve-se?"'
} else {
    Write-Host "`nsizes do not match -- rerun to resume."
    exit 1
}
