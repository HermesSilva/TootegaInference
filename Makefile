# TootegaEngine -- Makefile.  Ver docs/10-implementation.md.
#
# Um Makefile que um humano le: sem CMake, sem geracao, sem configure.
#
# Este e o caminho de build do Linux (gcc ou clang) -- musl e `-static` para o
# artefato CPU-only, glibc para o de GPU (docs/10, secao "Linux").
#
# No Windows o build e `build.ps1`, que usa o MSVC do Visual Studio.  As fontes
# sao C11 ISO e servem aos dois; nada de extensao GNU no nucleo.

CC     ?= gcc
ARCH   ?= x86-64-v3
BUILD  ?= build
SRC    := src
BIN    := tootega

# -march: NAO e `native`.  Um binario `native` gera SIGILL ao ser movido de
# maquina; o despacho para AVX-VNNI acontece em runtime.  Quem compila para si
# usa `make ARCH=native`.
#
# -ffp-contract=off: sem contracao a*b+c -> fma.  A contracao arredonda uma vez
# em vez de duas e portanto muda bits, e o criterio de saida do M1 e
# 32/32 tokens identicos ao oraculo (docs/15, D4).  A semantica do C escrito
# passa a ser exatamente a que se le; onde quisermos FMA, usamos o intrinseco.
#
# Sem -ffast-math, em lugar nenhum, por identico motivo.
CFLAGS := -std=c11 -O3 -Wall -Wextra -Wshadow -Wvla \
          -march=$(ARCH) -ffp-contract=off -fno-fast-math \
          -D_FORTIFY_SOURCE=2 -fstack-protector-strong \
          -MMD -MP

LDFLAGS := -static
LDLIBS  := -lm

ifeq ($(OS),Windows_NT)
  EXE     := .exe
  LDLIBS  += -lpsapi
else
  EXE     :=
  LDFLAGS += -Wl,-z,relro,-z,now
endif

SOURCES := $(wildcard $(SRC)/*.c)
OBJECTS := $(patsubst $(SRC)/%.c,$(BUILD)/%.o,$(SOURCES))
DEPS    := $(OBJECTS:.o=.d)
TARGET  := $(BIN)$(EXE)

.PHONY: all clean test selftest

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@
	@echo "  -> $@"

$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

# Os testes de kernel vivem dentro do binario (`tootega selftest`), nao num
# executavel separado: sao a definicao executavel dos kernels e tem de existir
# no artefato que roda, nao so no que se compila para testar.
test selftest: $(TARGET)
	./$(TARGET) selftest

clean:
	rm -rf $(BUILD) $(TARGET)

-include $(DEPS)
