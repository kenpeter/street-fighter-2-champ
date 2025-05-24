https://github.com/corbosiny/StreetFighterAI


https://github.com/linyiLYi/street-fighter-ai/tree/master




python -m retro.import .







python Lobby.py --epsilon 1.0 --rl 0.001



python Lobby.py --episodes 10 --epsilon 1.0 --rl 0.001 --resume



python Lobby.py --episodes 10 --epsilon 0.8 --rl 0.0005 --resume




All SF2 Characters as Save States

Added save state options for all 12 fighters in the game:
Keys 1-4: Ken, Ryu, Blanka, Chun-Li (original keys)
Keys 5-8: E.Honda, Guile, Dhalsim, Zangief
Keys 9-0: Balrog, Vega
Keys "-" and "=": Sagat, M.Bison (final bosses with highest difficulty)



| Training Phase       | Total Timesteps    | Epsilon (ε) | Learning Rate | Purpose                                                                 |
|----------------------|--------------------|-------------|---------------|-------------------------------------------------------------------------|
| Initial Exploration  | 0 - 100,000        | 1.0         | 0.001         | Broad exploration to discover basic strategies                         |
| Early Learning       | 100,000 - 250,000  | 0.7         | 0.0005        | Begin capitalizing on discovered patterns                              |
| Mid Learning         | 250,000 - 500,000  | 0.5         | 0.0003        | Balance exploration and exploitation                                   |
| Advanced Learning    | 500,000 - 750,000  | 0.3         | 0.0001        | Focus more on refining successful strategies                           |
| Fine-tuning          | 750,000 - 1,000,000| 0.1         | 0.00005       | Exploit learned knowledge with occasional exploration                  |
| Final Polish         | 1,000,000+         | 0.05        | 0.00001       | Almost pure exploitation for optimal performance                       |






python lobby.py --episodes 160 --epsilon 1.0 (20)





===



# Session 1 (already completed)
python Lobby.py --episodes 160 --epsilon 1.0

# Session 2
python Lobby.py --episodes 160 --epsilon 0.95 --resume

# Session 3
python Lobby.py --episodes 160 --epsilon 0.90 --resume

# Session 4
python Lobby.py --episodes 160 --epsilon 0.85 --resume

# Session 5
python Lobby.py --episodes 160 --epsilon 0.80 --resume

# Session 6
python Lobby.py --episodes 160 --epsilon 0.75 --resume

# Session 7
python Lobby.py --episodes 160 --epsilon 0.70 --resume

# Session 8
python Lobby.py --episodes 160 --epsilon 0.65 --resume

# Session 9
python Lobby.py --episodes 160 --epsilon 0.60 --resume

# Session 10
python Lobby.py --episodes 160 --epsilon 0.55 --resume

# Session 11
python Lobby.py --episodes 160 --epsilon 0.50 --resume

# Session 12
python Lobby.py --episodes 320 --epsilon 0.45 --resume

# Session 13
python Lobby.py --episodes 320 --epsilon 0.40 --resume

# Session 14
python Lobby.py --episodes 320 --epsilon 0.35 --resume

# Session 15
python Lobby.py --episodes 480 --epsilon 0.30 --resume

# Session 16
python Lobby.py --episodes 480 --epsilon 0.25 --resume

# Session 17
python Lobby.py --episodes 480 --epsilon 0.22 --resume

# Session 18
python Lobby.py --episodes 480 --epsilon 0.20 --resume

# Session 19
python Lobby.py --episodes 400 --epsilon 0.17 --resume

# Session 20
python Lobby.py --episodes 320 --epsilon 0.15 --resume

# Session 21
python Lobby.py --episodes 320 --epsilon 0.12 --resume

# Session 22
python Lobby.py --episodes 320 --epsilon 0.10 --resume

# Session 23
python Lobby.py --episodes 400 --epsilon 0.08 --resume

# Session 24
python Lobby.py --episodes 400 --epsilon 0.06 --resume

# Session 25
python Lobby.py --episodes 480 --epsilon 0.05 --resume

# Session 26
python Lobby.py --episodes 480 --epsilon 0.04 --resume

# Session 27
python Lobby.py --episodes 480 --epsilon 0.03 --resume

# Session 28
python Lobby.py --episodes 480 --epsilon 0.025 --resume

# Session 29
python Lobby.py --episodes 480 --epsilon 0.02 --resume

# Session 30
python Lobby.py --episodes 640 --epsilon 0.01 --resume






Street Fighter II AI Training Schedule
Phase 1: Exploration Phase (High Epsilon)
SessionEpisodesEpsilonLearning RateExpected Win RateTraining TimePurpose11601.00.0015-15%~1 minInitial exploration, random actions23200.950.00110-20%~2 minHeavy exploration, basic pattern learning34800.900.00115-25%~3 minDiscovering effective moves46400.850.00120-30%~4 minBuilding move sequences
Command Examples:

python lobby.py --episodes 160 --epsilon 1.0

python lobby.py --episodes 320 --epsilon 0.95 --resume

python lobby.py --episodes 480 --epsilon 0.90 --resume

python lobby.py --episodes 640 --epsilon 0.85 --resume





Phase 2: Balanced Learning (Medium Epsilon)
SessionEpisodesEpsilonLearning RateExpected Win RateTraining TimePurpose58000.800.00125-35%~5 minBalance exploration/exploitation69600.750.00130-40%~6 minRefining combat strategies711200.700.00135-45%~7 minLearning combo attacks812800.650.00140-50%~8 minDefensive positioning
Command Examples:
bashpython lobby.py --episodes 800 --epsilon 0.80 --resume
python lobby.py --episodes 960 --epsilon 0.75 --resume
python lobby.py --episodes 1120 --epsilon 0.70 --resume
python lobby.py --episodes 1280 --epsilon 0.65 --resume

Phase 3: Skill Development (Lower Epsilon)
SessionEpisodesEpsilonLearning RateExpected Win RateTraining TimePurpose914400.600.000845-55%~9 minAdvanced tactics1016000.550.000850-60%~10 minConsistent performance1117600.500.000855-65%~11 minStrategic depth1219200.450.000860-70%~12 minCompetitive level
Command Examples:
bashpython lobby.py --episodes 1440 --epsilon 0.60 --resume --rl 0.0008
python lobby.py --episodes 1600 --epsilon 0.55 --resume --rl 0.0008
python lobby.py --episodes 1760 --epsilon 0.50 --resume --rl 0.0008
python lobby.py --episodes 1920 --epsilon 0.45 --resume --rl 0.0008
Phase 4: Mastery Phase (Low Epsilon)
SessionEpisodesEpsilonLearning RateExpected Win RateTraining TimePurpose1320800.400.000665-75%~13 minNear-expert play1422400.350.000670-80%~14 minMastering special moves1524000.300.000675-85%~15 minProfessional-level tactics1625600.250.000680-90%~16 minTournament-ready
Command Examples:
bashpython lobby.py --episodes 2080 --epsilon 0.40 --resume --rl 0.0006
python lobby.py --episodes 2240 --epsilon 0.35 --resume --rl 0.0006
python lobby.py --episodes 2400 --epsilon 0.30 --resume --rl 0.0006
python lobby.py --episodes 2560 --epsilon 0.25 --resume --rl 0.0006
Phase 5: Expert Fine-tuning (Very Low Epsilon)
SessionEpisodesEpsilonLearning RateExpected Win RateTraining TimePurpose1727200.200.000485-95%~17 minExpert-level consistency1828800.150.000490-95%~18 minFrame-perfect execution1930400.100.000490-98%~19 minSuperhuman reflexes2032000.050.000495-99%~20 minPeak performance
Command Examples:
bashpython lobby.py --episodes 2720 --epsilon 0.20 --resume --rl 0.0004
python lobby.py --episodes 2880 --epsilon 0.15 --resume --rl 0.0004
python lobby.py --episodes 3040 --epsilon 0.10 --resume --rl 0.0004
python lobby.py --episodes 3200 --epsilon 0.05 --resume --rl 0.0004