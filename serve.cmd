@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem serve.cmd -- sobe o servidor e abre o chat no navegador.
rem
rem   serve.cmd                          padroes
rem   serve.cmd --port 9000 --ctx 32768  passa qualquer opcao adiante
rem   serve.cmd --help                   opcoes do engine
rem
rem O container e o tokenizer sao procurados; nada de caminho fixo. Para usar
rem outro modelo, defina TOOTEGA_MODEL antes de chamar.

set "EXE=%~dp0tootega.exe"
set "PORT=8080"
set "CTX=16384"

if "%~1"=="--help" goto :help
if "%~1"=="-h"     goto :help

if not exist "%EXE%" (
    echo.
    echo   %EXE% nao existe. Compile primeiro:
    echo.
    echo       .\build-all.ps1
    echo.
    exit /b 1
)

rem --- container -----------------------------------------------------------
rem Procura na ordem em que foram convertidos, do mais recente para o mais
rem antigo. TOOTEGA_MODEL tem precedencia sobre tudo.

if defined TOOTEGA_MODEL (
    set "MODEL=%TOOTEGA_MODEL%"
) else (
    set "MODEL="
    for %%M in (
        "models\glm47flash-v3\model.tootega"
        "models\glm47flash-v2\model.tootega"
        "models\glm47flash-int4\model.tootega"
    ) do (
        if not defined MODEL if exist "%%~M" set "MODEL=%%~M"
    )
)

if not defined MODEL (
    echo.
    echo   Nenhum container .tootega encontrado em models\.
    echo.
    echo   Baixe os pesos e converta:
    echo       python ptools\download_model.py
    echo       tools\convert.exe models\glm47flash-bf16 --out models\glm47flash-int4\model.tootega
    echo.
    exit /b 1
)
if not exist "%MODEL%" (
    echo   TOOTEGA_MODEL aponta para '%MODEL%', que nao existe.
    exit /b 1
)

rem --- tokenizer -----------------------------------------------------------
rem Vem do checkpoint original: o container guarda pesos, nao vocabulario.

set "TOK=models\glm47flash-bf16\tokenizer.json"
if not exist "%TOK%" (
    echo.
    echo   tokenizer.json nao encontrado em %TOK%
    echo   Ele vem do checkpoint BF16 original, nao do container.
    echo.
    exit /b 1
)

rem --- porta ---------------------------------------------------------------
rem Se a porta ja estiver ocupada, dizer isso agora vale mais do que deixar o
rem engine carregar 18 GB para so entao falhar no bind.

netstat -ano -p tcp | findstr /r /c:"LISTENING" | findstr /c:":%PORT% " >nul
if not errorlevel 1 (
    echo.
    echo   A porta %PORT% ja esta em uso -- outro tootega rodando?
    echo   Use: serve.cmd --port 9000
    echo.
    exit /b 1
)

rem --- navegador -----------------------------------------------------------
rem Aberto em paralelo, com atraso: o servidor le e aquece ~18 GB antes de
rem aceitar conexao, e uma aba que abre cedo demais mostra erro de conexao.

if not "%TOOTEGA_NO_BROWSER%"=="1" (
    start "tootega-browser" /min cmd /c "timeout /t 25 /nobreak >nul & start http://127.0.0.1:%PORT%/"
)

echo.
echo   modelo     %MODEL%
echo   tokenizer  %TOK%
echo   endereco   http://127.0.0.1:%PORT%/
echo.
echo   Aquecendo os pesos; o navegador abre sozinho em ~25 s.
echo   Ctrl-C encerra. A conversa nao sobrevive ao processo.
echo.

"%EXE%" serve "%MODEL%" --tokenizer "%TOK%" --port %PORT% --ctx %CTX% %*
exit /b %errorlevel%

:help
echo.
echo   serve.cmd -- sobe o servidor e abre o chat.
echo.
echo   serve.cmd [opcoes do engine]
echo.
echo   Variaveis:
echo     TOOTEGA_MODEL        caminho do container (default: o mais recente)
echo     TOOTEGA_NO_BROWSER   =1 para nao abrir o navegador
echo.
"%EXE%" serve
exit /b 0

