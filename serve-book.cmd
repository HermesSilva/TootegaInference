@echo off
setlocal
cd /d "%~dp0"

rem serve-book.cmd -- sobe o servidor com o container dos tres livros.
rem
rem   serve-book.cmd                     padroes do serve.cmd
rem   serve-book.cmd --port 9000         passa qualquer opcao adiante
rem
rem So muda o container: a porta, o log e as verificacoes sao os do `serve.cmd`,
rem que este script chama. O tokenizer e o mesmo do checkpoint original -- o
rem fine-tune nao mexe no vocabulario.
rem
rem   serve.cmd       o modelo limpo
rem   serve-book.cmd  o mesmo modelo, ajustado sobre a Cosmologia Composicional
rem
rem O PROMPT VAI LIMPO. Nada e injetado nele: a pergunta do usuario chega ao
rem modelo como ele a escreveu. Foi decisao do Hermes, e e de produto -- o
rem modelo deve SER o especialista, nao um leitor com cola no prompt.
rem
rem O preco esta medido e nao e pequeno: este container foi treinado com RAFT,
rem que troca memorizacao por leitura de proposito. Sem trecho no contexto,
rem 0 de 11 respostas revisadas a mao estavam doutrinariamente corretas, contra
rem 26 de 29 com trecho (docs/22). Ele diz que a Arena e um estadio de futebol.
rem
rem   O conserto NAO e reinjetar texto -- e um treino que ponha o conteudo nos
rem   pesos. Ver docs/13 D17.
rem
rem A busca continua no engine e desligada por padrao. Para liga-la de novo,
rem explicitamente, numa sessao:
rem
rem   .\tools\index.exe build "models\Versao Completa" --out models\v5-sft\cosmologia.ttindex
rem   .\serve-book.cmd --index models\v5-sft\cosmologia.ttindex

set "MODEL=%~dp0models\v5-sft\glm47flash-v5-sft.tootega"

if not exist "%MODEL%" (
    echo.
    echo   Container dos livros nao encontrado:
    echo     %MODEL%
    echo.
    echo   Baixe com:
    echo       .\finetune\download-model.ps1
    echo.
    exit /b 1
)

rem `--think off` porque todo par de treino do RAFT fechava `</think>` na hora.
rem Com o raciocinio aberto o modelo cai num rascunho em ingles deliberando
rem sobre uma pergunta em portugues -- e uma propriedade do container, nao da
rem busca, entao vai dito em voz alta.
set "TOOTEGA_MODEL=%MODEL%"
call "%~dp0serve.cmd" --think off %*
exit /b %errorlevel%
