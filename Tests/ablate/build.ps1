# Delega para o build compartilhado -- ver tools/shared/build-tool.ps1.
& (Join-Path $PSScriptRoot '..\..\tools\shared\build-tool.ps1') -Name 'ablate' -Dir $PSScriptRoot @args
