1. epizoda za blokirana stoji na pocetku, pocetna tacka je bio dropoff.  (forced stop)
2. epizoda krece se levo desno, prolazi kroz pickoff, ali ne pokuplja putnika
3. epizoda uspesna
4. epizoda samo stoji u mestu (forced stop)
5. ep takodje samo stoji u mestu, na mestu gde moze biti pickoff, ali to nije sada slucaj (forced stop)
6. ep stoji na mestu gde treba da bude dropoff i ne pomera se (forced stop)
7. ep uspesna
8. ep uspesna
9. ep vrti se u krug i dolazi na mesto gde je pickoff, ali sada tu ne treba da bude pickoff. (forced stop)
10. ep prolazi kroz pickoff ali ne pokuplja putnika (forced stop)
11. ep uspesna
12. ep vrti se u krug i dolazi na mesto gde je pickoff, ali sada tu ne treba da bude pickoff. (forced stop)
13. ep pravi nekoliko koraka, nakon cega staje skroz (forced stop)
14. ep stoji u mestu, ne pomera se (forced stop)
15. ep stoji na mestu gde moze da bude potencijalni pickoff i ne pomera se (forced stop)




ONLINE Q-Learning

=== Evaluation ===
Agent        : q_learning
Model        : models/online/q_learning.pkl
Episodes     : 2000
Avg reward   : 8.02
Std reward   : 2.61
Avg steps    : 12.98
Success rate : 100.00%

Offline Q-Learning (random dataset)

=== Evaluation ===
Agent        : q_learning
Model        : models/offline/q_learning_random.pkl
Episodes     : 2000
Avg reward   : 8.02
Std reward   : 2.61
Avg steps    : 12.98
Success rate : 100.00%

Offline Q-learning (behaviour dataset, eps=0.1)

=== Evaluation ===
Agent        : q_learning
Model        : models/offline/q_learning_behavior_q_eps01.pkl
Episodes     : 2000
Avg reward   : 7.20
Std reward   : 13.39
Avg steps    : 13.72
Success rate : 99.60%


Final sweep eval

Online Q-learning (sweep)

=== Evaluation (START-STATE SWEEP) ===
Agent        : q_learning
Model        : models/online/q_learning.pkl
States       : 500
Avg reward   : -32.16
Std reward   : 84.00
Avg steps    : 48.96
Success rate : 61.40%

Offline Q-learning random (sweep)

=== Evaluation (START-STATE SWEEP) ===
Agent        : q_learning
Model        : models/offline/q_learning_random.pkl
States       : 500
Avg reward   : -33.02
Std reward   : 84.61
Avg steps    : 49.73
Success rate : 61.00%

Offline Q-learning behaviour (sweep)

=== Evaluation (START-STATE SWEEP) ===
Agent        : q_learning
Model        : models/offline/q_learning_behavior_q_eps01.pkl
States       : 500
Avg reward   : -37.28
Std reward   : 87.50
Avg steps    : 53.58
Success rate : 59.20%





Online SARSA

=== Evaluation ===
Agent        : sarsa
Model        : models/online/sarsa.pkl
Episodes     : 2000
Avg reward   : 7.97
Std reward   : 2.64
Avg steps    : 13.03
Success rate : 100.00%

Offline SARSA (random dataset)

=== Evaluation ===
Agent        : sarsa
Model        : models/offline/sarsa_random.pkl
Episodes     : 2000
Avg reward   : -412.45
Std reward   : 656.36
Avg steps    : 156.88
Success rate : 22.75%


Offline SARSA (behaviour dataset, eps=0.1)

=== Evaluation ===
Agent        : sarsa
Model        : models/offline/sarsa_behavior_sarsa_eps01_big.pkl
Episodes     : 2000
Avg reward   : -101.31
Std reward   : 300.21
Avg steps    : 75.67
Success rate : 66.00%

Final sweep evaluacija


Online SARSA (sweep)

=== Evaluation (START-STATE SWEEP) ===
Agent        : sarsa
Model        : models/online/sarsa.pkl
States       : 500
Avg reward   : -32.62
Std reward   : 84.29
Avg steps    : 49.38
Success rate : 60.60%

Offline SARSA (random, sweep)

=== Evaluation (START-STATE SWEEP) ===
Agent        : sarsa
Model        : models/offline/sarsa_random.pkl
States       : 500
Avg reward   : -298.63
Std reward   : 533.74
Avg steps    : 146.89
Success rate : 26.80%

Offline SARSA (behaviour, sweep)

=== Evaluation (START-STATE SWEEP) ===
Agent        : sarsa
Model        : models/offline/sarsa_behavior_sarsa_eps01_big.pkl
States       : 500
Avg reward   : -111.42
Std reward   : 275.28
Avg steps    : 91.11
Success rate : 56.00%



U RL postoje 2 stvari:

1. Policy (pi) - kako biramo akciju
2. Value function (Q) - koliko je dobra akcija u stanju

U tabularnom RL:

pi(s) = argmax Q(s,a)

On-policy algoritam

Uci vrednosti politike koju trenutno koristi za ponasanje

Uci Q^pi(s,a)

To znaci:
- Evaluira sopstvenu politiku
- Uci ono sto stvarno radi

Primer: SARSA

Off-policy algoritam

Uci vrednosti optimalne politike bez obzira na to kojom politikom su podaci generisani

Uci Q^*(s,a)

Primer: Q-learning  

SARSA Update (on-policy)

Q(s,a) <- r+gama*Q(s',a')

- koristi sledecu akciju koju politika stvarno bira
- uci vrednosti te konkretne  politike

Ako je politika randon -> uci random
Ako je politika eps-greedy -> uci eps greedy

Q-learning update (off-policy)

Q(s,a) <- r + gama maxQ(s',a')

- koristi najbolju mogucu akciju
- ignorise koju je akciju behaviour politika stvarno izabrala

Zbog toga moze da uci optimalnu politiku iako dataset dolazi iz random politike

On-policy uci vrednosti trenutnog ponasanja (evaluira svoju politiku)

Off-policy uci vrednosti najboljeg ponasanja (poboljsava politiku), cak i iz tudjih podataka

SARSA: target zavisi od sledece akcije koju ces stvarno uzeti
Q-learning: target zavisi od najbolju moguce sledece akcije



Tvoj projekat je organizovan po veoma dobroj logici:

src/taxi_rl/ sadrži glavnu logiku sistema

scripts/ sadrži ulazne tačke za pokretanje eksperimenata

data/ sadrži dataset-e

models/ sadrži istrenirane modele

results/ je rezervisan za tabele i grafike rezultata

configs/ sadrži parametre eksperimenta

To znači da si projekat podelio na:

implementaciju

pokretanje

podatke

rezultate

To je jako dobra praksa, jer sprečava da ti se sve pomeša u jednom folderu.


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


Tok A — Online trening

    Pokrećeš scripts/train_online.py

    On učitava config i argumente

    Pravi okruženje

    Kreira agenta (Q-learning ili SARSA)

    Poziva src/taxi_rl/online/trainer.py

    Trening prolazi epizodu po epizodu

    Model se snima u models/online/

Tok B — Generisanje offline dataset-a

    Pokrećeš scripts/collect_offline_dataset.py

    Pravi se env

    Bira se polisa ponašanja:

        random

        behavior

    Agent prolazi kroz epizode

    Tranzicije se zapisuju u CSV

    Dataset se čuva u data/offline/

Tok C — Offline trening

    Pokrećeš scripts/train_offline.py

    Učitava se CSV dataset

    Kreira se agent

    Poziva se src/taxi_rl/offline/trainer.py

    Model prolazi više epoha kroz dataset

    Novi model se čuva u models/offline/

Tok D — Evaluacija i demo
    Evaluacija

        Pokrećeš scripts/evaluate.py

        Učitavaš model

        Pokrećeš standard eval ili sweep

        Dobijaš metrike

    Demo

        Pokrećeš scripts/demo.py

        Učitavaš model

        Gledaš ponašanje modela epizodu po epizodu kroz GUI
