# Delega para o build compartilhado -- ver tools/shared/build-tool.ps1.
& (Join-Path $PSScriptRoot '..\shared\build-tool.ps1') -Name 'agent' -Dir $PSScriptRoot @args
