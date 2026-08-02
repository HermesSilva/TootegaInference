# Delegates to the shared build -- see tools/shared/build-tool.ps1.
& (Join-Path $PSScriptRoot '..\shared\build-tool.ps1') -Name 'index' -Dir $PSScriptRoot @args
