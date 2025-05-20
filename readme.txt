https://github.com/corbosiny/StreetFighterAI


https://github.com/linyiLYi/street-fighter-ai/tree/master




python -m retro.import .








python Lobby.py --episodes 100 --epsilon 1.0 --rl 0.001



python Lobby.py --episodes 1 --epsilon 0.7 --rl 0.0005 --resume




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




