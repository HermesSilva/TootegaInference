@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem serve.cmd -- sobe o servidor e abre o chat no navegador.
rem
rem   serve.cmd                          padroes
rem   serve.cmd --port 9000 --ctx 32768  passa qualquer opcao adiante
rem   serve.cmd --verbose                mostra o que entra e sai no console
rem   serve.cmd --help                   opcoes do engine
rem
rem O container e o tokenizer sao procurados; nada de caminho fixo. Para usar
rem outro modelo, defina TOOTEGA_MODEL antes de chamar.

set "EXE=%~dp0tootega.exe"
set "PORT=8080"
rem 150.000 posicoes: 16,2 GB de KV sobre os 18,5 GB de pesos, deixando ~15 GB
rem livres nesta maquina. E o teto de MEMORIA -- o do modelo e 202.752.
rem
rem Nao e o teto util: a 11 tok/s de prefill, encher 150k levaria 3,8 h. Serve
rem para nunca esbarrar no limite, nao para ser preenchido.
set "CTX=150000"

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

rem O navegador NAO e aberto: o servidor tambem atende o Cockpit e clientes
rem OpenAI, e abrir uma aba a forca atrapalha quem nao quer a UI web. Para
rem usa-la, abra http://127.0.0.1:PORTA/ a mao.

echo.
echo   modelo     %MODEL%
echo   tokenizer  %TOK%
echo   contexto   %CTX% posicoes
echo   endereco   http://127.0.0.1:%PORT%/
echo.
echo   Aquecendo os pesos; leva ~25 s antes do primeiro pedido.
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
echo.
echo   A UI web fica em http://127.0.0.1:8080/ -- abra a mao se quiser.
echo.
"%EXE%" serve
exit /b 0

