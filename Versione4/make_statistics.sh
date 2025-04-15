#!/bin/bash

# Lista dei valori da testare per len_rolling
rolling_windows=(25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450)

# Crea la cartella dei log se non esiste
mkdir -p logs

# Loop su ogni valore e lancio dello script Python
for len in "${rolling_windows[@]}"
do
    echo "Eseguo con len_rolling=$len"

    python prova.py --len_rolling $len \
        > logs/output_len${len}.log \
        2> logs/error_len${len}.log

    # Se vuoi vedere anche un messaggio a schermo in caso di errore
    if [ $? -ne 0 ]; then
        echo "❌ Errore con len_rolling=$len — controlla logs/error_len${len}.log"
    else
        echo "✅ Completato len_rolling=$len"
    fi
done