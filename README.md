configs - cuva konfiguracione fajlove

data - folder cuva podatke. U data/offline se nalaze CSV datasetovi koji se koriste za offline treniranje. Ovi fajlovi predstavljaju snimljene tranzicije iz okruzenja


models - ovde se cuvaju istrenirani modeli
models/offline - modeli trenirani iz dataseta 
models/online - modeli istrenirani direktno iz interakcije sa okruzenjem

scripts - ovde se nalaze skripte koje korisnik pokrece. Ove skripte uglavnom ucitavaju argumente, zovu finkcije iz src/taxi_rl i pokrecu odredjeni deo pipeline-a. To znaci da su one interfejs izmedju korisnika i glavne implementacije


src/taxi_rl -  srce projekta, cela implementacija sistema

agents - ovde su implementirani algoritmi q-learning i sarsa

online - logika online treniranja (kako agent prolazi kroz epizodu, kako se update radi dok interaguje sa okruzenjem)

offline - logika offline dela (ucitavanje dataseta i offline update kroz postojece tranzicije)

eval - ovde je evaulacija (standard eval, sweep eval, success rate, metrike)

env.py - apstrakcija nad Taxi okruzenjem: kreiranje env, reset, step, kompatibilnost gym

policies.py - ovde su pravila izbora akcije (random, e-greedy)

utils.py - pomocne funkcije (YAML, pickle, seed, schedule)


Online trening

    Pokrećeš scripts/train_online.py

    On učitava config i argumente

    Pravi okruženje

    Kreira agenta (Q-learning ili SARSA)

    Poziva src/taxi_rl/online/trainer.py

    Trening prolazi epizodu po epizodu

    Model se snima u models/online/

Generisanje offline dataset-a

    Pokrećeš scripts/collect_offline_dataset.py

    Pravi se env

    Bira se polisa ponašanja:

        random

        behavior

    Agent prolazi kroz epizode

    Tranzicije se zapisuju u CSV

    Dataset se čuva u data/offline/

Offline trening

    Pokrećeš scripts/train_offline.py

    Učitava se CSV dataset

    Kreira se agent

    Poziva se src/taxi_rl/offline/trainer.py

    Model prolazi više epoha kroz dataset

    Novi model se čuva u models/offline/

Evaluacija i demo
    
    Evaluacija

        Pokrećeš scripts/evaluate.py

        Učitavaš model

        Pokrećeš standard eval ili sweep

        Dobijaš metrike

    Demo

        Pokrećeš scripts/demo.py

        Učitavaš model

        Gledaš ponašanje modela epizodu po epizodu kroz GUI
