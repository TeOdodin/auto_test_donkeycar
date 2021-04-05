import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'config'))

import gym
import gym_donkeycar
import numpy as np
from donkeycar.utils import get_model_by_type
from donkeycar.parts.camera import PiCamera
import config as cfg
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
exe_path = "../sdsandbox/outputs_win/donkey_sim.exe"
port = 8887

conf = { "exe_path" : exe_path, "port" : port, "host" : "localhost" }
# cfg = {"DEFAULT_MODEL_TYPE": "linear", "IMAGE_H": 120, "IMAGE_W": 160, "IMAGE_DEPTH": 3}

scenes = ["donkey-generated-track-v0","donkey-waveshare-v0","donkey-warehouse-v0"]

def auto_test(model_type, model_path, dataset):
  # PLAY
  model = get_model_by_type(model_type, cfg)
  model.load(model_path)
  # cam = PiCamera(image_w=cfg["IMAGE_W"], image_h=cfg["IMAGE_H"], image_d=cfg["IMAGE_DEPTH"], framerate=20, vflip=False, hflip=False)

  colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,255,0), (100,50,200), (200,100,50)]
  colornames = ["g-", "b-", "r-", "c-", "m-", "y-"]

  tries = []

  for scene in scenes:
    env = gym.make(scene, conf=conf)
    print(scene)
    for k in range(1, 7):
      print(f"Round {k}")
      obs = env.reset()
      speed = 0

      Xs = []
      Ys = []

      for t in range(200):
        # frame = cam.run()
        # action = np.array(model.run(frame)) # drive straight with small spee
        inference = model.run(obs)
        speed = speed + inference[1]
        action = np.array([inference[0], speed])
        # print(action)
        # execute the action
        obs, reward, done, info = env.step(action)
        # print(info["pos"])
        Xs.append(info["pos"][0])
        Ys.append(info["pos"][2])
        # print(obs)
      
      tries.append([scene, k, Xs, Ys])

    # Exit the scene
    env.close()

  coeffs = {
    "donkey-generated-track-v0": [3.5, -3.5, 80, 360],
    "donkey-waveshare-v0": [14, 14, 270, 160],
    "donkey-warehouse-v0": [-3.5, 3.5, 380, 125]
  }

  lastTry = tries[0][0]
  img = cv2.imread(f"images/{tries[0][0]}.png")
  plt.figure(1)

  for n in range(len(tries)):
    inst = tries[n]
    if lastTry != inst[0]:
      cv2.imwrite(f"outputs/{model_type}/{dataset}/track_{lastTry}.png", img)
      plt.savefig(Path(f"outputs/{model_type}/{dataset}/pos_{lastTry}").with_suffix('.png'))
      plt.clf()
      plt.figure(1)
      lastTry = inst[0]
      img = cv2.imread(f"images/{inst[0]}.png")
    plt.plot(Xs, Ys, colornames[n%len(colornames)])

    Xs = inst[2]
    Ys = inst[3]
    lastX = Xs[0]
    lastY = Ys[0]
    offset = coeffs[lastTry]
    for i in range(1, len(Xs)):
      rX = round(Xs[i] * offset[0] + offset[2])
      rY = round(Ys[i] * offset[1] + offset[3])
      cv2.circle(img, (rX, rY), 2, colors[inst[1]%10], 1)

    import json
    with open(Path(f"outputs/{model_type}/{dataset}/pos_{inst[0]}_{inst[1]}").with_suffix('.json'), 'w') as outfile:
        json.dump({"Xs": Xs, "Ys": Ys}, outfile)

  cv2.imwrite(f"outputs/{model_type}/{dataset}/track_{lastTry}.png", img)
  plt.savefig(Path(f"outputs/{model_type}/{dataset}/pos_{lastTry}").with_suffix('.png'))