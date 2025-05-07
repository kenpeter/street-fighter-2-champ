https://github.com/corbosiny/StreetFighterAI


https://github.com/linyiLYi/street-fighter-ai/tree/master




python -m retro.import .






python Lobby.py --episodes 2 --background --resume



# Install Docker and NVIDIA Container Toolkit
chmod +x setup_tensorflow_cuda.sh
./setup_tensorflow_cuda.sh

# Build Docker image
docker build -t street-fighter-ai .

# Run training with GPU support
docker run --gpus all street-fighter-ai -e 20 -b






All SF2 Characters as Save States

Added save state options for all 12 fighters in the game:
Keys 1-4: Ken, Ryu, Blanka, Chun-Li (original keys)
Keys 5-8: E.Honda, Guile, Dhalsim, Zangief
Keys 9-0: Balrog, Vega
Keys "-" and "=": Sagat, M.Bison (final bosses with highest difficulty)



How to Create Higher-Level AI States: