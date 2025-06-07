from gymnasium.envs.registration import register

def setup():
    register(
        id='Connect4-v0',
        entry_point='gym_env:Connect4Env',
    )
    register(
        id='Connect4Bitboard-v0',
        entry_point='gym_env_2:Connect4Env',
    )