https://github.com/corbosiny/StreetFighterAI


https://github.com/linyiLYi/street-fighter-ai/tree/master




python -m retro.import .








60%
python Lobby.py --episodes 1 --background --epsilon 0.9 --resume



python Lobby.py --play




All SF2 Characters as Save States

Added save state options for all 12 fighters in the game:
Keys 1-4: Ken, Ryu, Blanka, Chun-Li (original keys)
Keys 5-8: E.Honda, Guile, Dhalsim, Zangief
Keys 9-0: Balrog, Vega
Keys "-" and "=": Sagat, M.Bison (final bosses with highest difficulty)



How to Create Higher-Level AI States:




| Win Rate | Epsilon | Learning Rate | Total Timesteps |
|----------|---------|---------------|-----------------|
| 30%      | 0.90    | 0.002         | 20,000          |
| 40%      | 0.70    | 0.0015        | 50,000          |
| 50%      | 0.50    | 0.001         | 80,000          |
| 60%      | 0.30    | 0.001         | 100,000         |
| 70%      | 0.20    | 0.0005        | 250,000         |
| 80%      | 0.10    | 0.0001        | 500,000         |
| 90%      | 0.05    | 0.00005       | 1,000,000       |





Win Percentage	Estimated Training Steps
50%	50,000
60%	100,000
70%	200,000
80%	500,000
90%	1,000,000
