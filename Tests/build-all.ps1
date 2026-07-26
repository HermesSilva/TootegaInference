# Compila todas as ferramentas de teste.
Get-ChildItem $PSScriptRoot -Directory | ForEach-Object {
    $b = Join-Path $_.FullName 'build.ps1'
    if (Test-Path $b) { & $b @args }
}
