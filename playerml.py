import asyncio

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env

import time

from playermaxdamage import MaxDamagePlayer

import torch.nn as nn
from ml import Agent

from tabulate import tabulate
import matplotlib.pyplot as plt

from poke_env import gen_data
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.player import Player
from poke_env.player.battle_order import BattleOrder

from poke_env.player import (
    SinglesEnv,
    SingleAgentWrapper,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
)


class SimpleRLPlayer(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        self.observation_spaces = {
            agent: Box(
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
    def calc_reward(self, current_battle) -> float:
        #not deterministic?
       
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )
    
    #This is causing the non-deterministic error
    def embed_battle(self, battle: AbstractBattle)  -> np.ndarray[np.float32]:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.id != "curse":
                
            
                if move.type:
                    type_chart = gen_data.GenData.from_gen(4).type_chart
                    moves_dmg_multiplier[i] = move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                        type_chart=type_chart
                    )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return np.float32(low)
        #return np.float32(final_vector)

    #def describe_embedding(self) -> Space:
    #    low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
    #    high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
    #    return Box(
    #        np.array(low, dtype=np.float32),
    #        np.array(high, dtype=np.float32),
    #        dtype=np.float32,
    #    )
    
class SinglesTestEnv(SinglesEnv[npt.NDArray[np.float32]]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(np.array([0, 0]), np.array([6, 6]), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.strict = False

    def calc_reward(self, battle) -> float:
        return 0.0#self.reward_computing_helper(battle)

    def embed_battle(self, battle: AbstractBattle):
        to_embed = []
        fainted_mons = 0
        for mon in battle.team.values():
            if mon.fainted:
                fainted_mons += 1
        to_embed.append(fainted_mons)
        fainted_enemy_mons = 0
        for mon in battle.opponent_team.values():
            if mon.fainted:
                fainted_enemy_mons += 1
        to_embed.append(fainted_enemy_mons)

        #print(self.observation_space)
        #print(self.reset())
        return np.array([0,0], dtype=np.float32)#np.array(to_embed)
    
class SinglesTestEnv2(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(np.array([0, 0]), np.array([6, 6]), dtype=np.float32)
            for agent in self.possible_agents
        }

    def calc_reward(self, battle) -> float:
        return 0.0

    def embed_battle(self, battle):
        return np.array([0,0], dtype=np.float32)
    
    
class PokeAgent(nn.Module):
    def __init__(self, vocab_size, num_choices):
        super(PokeAgent, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, 32)
        self.linear = nn.Linear(32, 32)
        self.activation = nn.ReLU()
        self.out = nn.Linear(32, num_choices)
    
    def forward(self, turn_input):
        embedding = self.embedding(turn_input)
        embedding = embedding.mean(dim=0)
        
        x = self.linear(embedding)
        x = self.activation(x)
        x = self.out(x)
        
        return x
    
async def main():
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_player = SimpleRLPlayer(
        battle_format="gen4randombattle", start_challenging=True, strict=False
    )
    test_player2 = SinglesTestEnv(
            battle_format=f"gen8randombattle",
            log_level=25,
            start_challenging=True,
            strict=False,
        )
    test_env = SingleAgentWrapper(test_player, MaxDamagePlayer())
   
    
    #test_out = test_env.reset()
    
    check_env(test_env)
    test_env.close()

    opponent = RandomPlayer(battle_format="gen4randombattle")
    train_player = SimpleRLPlayer(
        battle_format="gen4randombattle", start_challenging=True
    )
    train_env = train_player
    opponent = MaxDamagePlayer(battle_format="gen4randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen4randombattle", start_challenging=True
    )
    eval_env = eval_env

    n_action = train_env.action_space
    input_shape = (1,) + train_env.observation_spaces['SimpleRLPlayer 1'].shape

    dqn = Agent(train_env)
    start = time.time()
    records = dqn.train(1000)

    

    

    print(
        "Learning player won %d / 100 battles [this took %f seconds]"
        % (
            train_env.n_won_battles, time.time() - start
        )
    )

    print("Results against random player:")
    dqn.test(eval_env, 100)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )

    #plt.plot(records)
    #plt.show()

    print("here")


    player1 = RandomPlayer()
    player2 = RandomPlayer()

    await player1.battle_against(player2, n_battles=1)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
    #main()