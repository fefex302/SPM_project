# ==========================================
#  MAKEFILE PER RANDOM FOREST (SPM PROJECT)
# ==========================================

# 1. Il Compilatore
CXX = g++

# 2. Le opzioni di compilazione (Flags)
# -std=c++17   : Usa lo standard moderno C++17
# -O3          : Massima ottimizzazione (FONDAMENTALE per lo speedup)
# -march=native: Ottimizza per la CPU specifica del cluster/PC
# -Wall        : Mostra tutti gli avvisi (per evitare errori stupidi)
# -Iinclude    : Dice al compilatore di cercare i file .h nella cartella 'include'
# -pthread     : Necessario per i thread (anche se non li usi ancora, serve per il futuro)
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Iinclude -pthread

# 3. Nome dell'eseguibile finale
TARGET = RandomForest

# 4. Lista dei file sorgente (.cpp)
# Elenchiamo manualmente i file per essere sicuri dell'ordine e della posizione
SRCS = main.cpp \
       src/Data.cpp \
       src/Tree.cpp \
       src/RandomForest.cpp

# 5. Trasformiamo la lista dei .cpp in una lista di .o (File Oggetto)
# Questa è una sostituzione automatica di stringa
OBJS = $(SRCS:.cpp=.o)

# ==========================================
#  REGOLE (Cosa deve fare il make)
# ==========================================

# Regola di default (quella che parte se scrivi solo 'make')
all: $(TARGET)

# Regola per creare l'eseguibile finale (LINKING)
# Unisce tutti i file oggetto (.o) in un unico programma
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Regola generica per compilare i file .cpp in .o (COMPILAZIONE)
# $< è il file sorgente (.cpp)
# $@ è il file destinazione (.o)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regola per pulire tutto (utile se cambi flags o fai casino)
# Si lancia con: make clean
clean:
	rm -f $(OBJS) $(TARGET)
	rm -f src/*.o  # Rimuove anche gli oggetti nella sottocartella per sicurezza

# Regola 'phony' per evitare conflitti se hai file che si chiamano 'clean' o 'all'
.PHONY: all clean