# Delegates to the shared build -- see tools/shared/build-tool.ps1.
& (Join-Path $PSScriptRoot '..\..\tools\shared\build-tool.ps1') -Name 'mmbatch' -Dir $PSScriptRoot @args
