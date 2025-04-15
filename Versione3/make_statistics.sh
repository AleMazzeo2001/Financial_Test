#!/bin/bash

# Lista dei valori da testare per len_rolling
rolling_windows=(25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450)
 

# Loop su ogni valore e lancio dello script Python
for len in "${rolling_windows[@]}"
do
    echo "Eseguo con len_rolling=$len"
    python prova.py --len_rolling $len
done