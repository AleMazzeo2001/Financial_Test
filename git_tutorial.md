TUTORIAL: https://www.w3schools.com/git/

Elenco di comandi git:

# Config Global User
   ° git config --global user.name "AleMazzeo2001"
   ° git config --global user.email "alessandro.mazzeo01@universitadipavia.it"

# Get Started
   ° git init #inizia la repo di git
   ° git status #controlla lo status della repo
   ° git status --short #informazioni più sintetiche
   ° git add financial_test.py #aggiunge il file al branch
   ° git add -A #aggiunge tutti i file presenti
   ° git commit -m "Primo commit su Git" #manda il branch nella repo: buona norma mettere un messaggio
   ° git commit -a -m "Messaggio di commit" #aggiunge tutte le modifiche ai file già tracciati (precedentemente  add) + il commit
   ° git log #vedo la storia

# Chiedere Aiuto 
   ° git command -help
   ° git help --all #lista di tutti i comandi

# Branch
   ° git branch versione-parallela # creo un nuovo branch
   ° git checkout versione-parallela # passo al nuovo brunch
   ° git checkout -b emergency-fix #crea nuovo branch e ci passa in automatico

# Riunire il Branch al Main
   ° git checkout main
   ° git merge emergency-fix

# Push Local Repository on GitHub
   ° crea per prima cosa la repository su GitHub
   ° git remote add origin https://github.com/AleMazzeo2001/Financial_Test.git
   ° git push --set-upstream origin main #aggiungo la repository su GitHib


