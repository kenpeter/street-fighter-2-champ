https://github.com/corbosiny/StreetFighterAI


https://github.com/linyiLYi/street-fighter-ai/tree/master




python -m retro.import .






python Lobby.py --episodes 10 --background --resume --epsilon 0.8






All SF2 Characters as Save States

Added save state options for all 12 fighters in the game:
Keys 1-4: Ken, Ryu, Blanka, Chun-Li (original keys)
Keys 5-8: E.Honda, Guile, Dhalsim, Zangief
Keys 9-0: Balrog, Vega
Keys "-" and "=": Sagat, M.Bison (final bosses with highest difficulty)



How to Create Higher-Level AI States:






Step 1900, Player health: 151, Enemy health: 151
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
STATE LOG [2000]: {'step': 2000, 'health': 100, 'enemy_health': 148, 'status': 512, 'enemy_status': 512, 'x_position': 100, 'enemy_x_position': 200, 'done_flag': False}
Step 2000, Player health: 100, Enemy health: 148
Episode result: LOSS
Episode steps: 2000
Episode reward: 2100.0
Total steps so far: 312837
WARNING: Episode terminated due to step limit, not game completion

======= TRAINING SUMMARY =======
Total training timesteps: 2099888
Total episodes completed: 7
Total training time: 1704.05 seconds
Final average reward: 3.4147
Reward improvement: +0.9493
Final average loss: 0.329847
Loss improvement: -0.093902
Learning status: NEGATIVE - Agent may be stuck in suboptimal policy
=================================


========= TRAINING SESSION SUMMARY =========
Total training steps: 312837
Total episodes: 140
Total Win/Loss Record: 38W - 102L (27.14%)
Current session record: 38W - 102L (3837.25%)
Accumulated stats: Yes (--resume)
Total training time: 1703.60 seconds
Training efficiency: 183.63 steps/second
Agent's accumulated training timesteps: 2099888
Reward trend: -78.96% change
Learning assessment: NEGATIVE - Agent may be stuck in suboptimal policy
===========================================
(sf2) kenpeter@kenpeter-ubuntu:~/work/street-fighter-2-champ$ 

