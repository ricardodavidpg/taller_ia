import numpy as np
import matplotlib.pyplot as plt
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import (
    TransformReward,
    RecordVideo,
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
    AtariPreprocessing,
    FrameStackObservation
)

def show_observation(observation):
    dimension = observation.shape
    if len(dimension) == 3:
        if dimension[2] == 3:
            plt.imshow(observation)
        elif dimension[2] == 1:
            plt.imshow(observation[:, :, 0], cmap='gray')
    elif len(dimension) == 2:
        plt.imshow(observation, cmap='gray')
    else:
        raise ValueError("Invalid observation shape")
    plt.show()
    
def show_observation_stack(observation):
    frames = observation.shape[0]
    for i in range(frames):
        show_observation(observation[i])


class FireOnLifeLostWrapper(gymnasium.Wrapper):
    """Presiona FIRE automáticamente tras reset y tras cada pérdida de vida."""
    def __init__(self, env):
        super().__init__(env)
        self._prev_lives = None

    def reset(self, **kwargs):
        # 1) Reset normal
        obs, info = self.env.reset(**kwargs)
        # 2) Inyectar FIRE para arrancar la partida
        obs, _, terminated, truncated, info = self.env.step(1)
        # Si por alguna razón el juego acabó (raro), reinicia otra vez
        if terminated or truncated:
            return self.reset(**kwargs)
        # 3) Guarda el número de vidas inicial
        self._prev_lives = info.get('lives')
        return obs, info

    def step(self, action):
        # 1) Paso normal del agente
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 2) Detecta pérdida de vida
        current_lives = info.get('lives', self._prev_lives)
        if (current_lives < self._prev_lives) and not (terminated or truncated):
            # 3) Inyecta FIRE para reanudar tras perder vida
            obs, fire_reward, terminated, truncated, info = self.env.step(1)
            reward += fire_reward  # opcional: sumar recompensa de FIRE
        # 4) Actualiza contador de vidas
        self._prev_lives = current_lives
        return obs, reward, terminated, truncated, info

def make_env(
    env_name: str,
    render_mode: str = "rgb_array",
    # Video
    video_folder: str | None = "./videos",
    name_prefix: str = "",
    record_every: int | None = None,
    # Preprocesado
    grayscale: bool = False,
    screen_size: int = 84,
    stack_frames: int = 4,
    skip_frames: int = 4
) -> gymnasium.Env:

    env = gymnasium.make(env_name, render_mode=render_mode, frameskip=1)
    
    if video_folder is not None and record_every is not None:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda ep: ep % record_every == 0,
            fps=env.metadata.get("render_fps", 30) * skip_frames,
        )
    
    # env = FireOnLifeLostWrapper(env)
    
    env = AtariPreprocessing(
        env,
        noop_max=10,
        frame_skip=skip_frames,
        screen_size=screen_size,
        grayscale_obs=grayscale,
        grayscale_newaxis=False
    )
    
    # stack frames
    env = FrameStackObservation(env, stack_size=stack_frames)
    
    # clip rewards
    sign_fn = lambda r: 1 if r > 0 else (-1 if r < 0 else 0)
    env = TransformReward(env, sign_fn)
    
    return env
