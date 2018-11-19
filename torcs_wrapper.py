from torcs import Torcs
import cv2
import gym
import numpy as np

class TorcsWrapper:
    def __init__(self, port=3101, noisy=True, throttle=0.20, control_dim = 1, k = 2.0):
        self.episode = 0
        self.step_per_episode = 0
        self.step_total = 0
        self.throttle = throttle
        self.control_dim = control_dim
        self.k = k
        self.env = Torcs(vision=True, port=port, noisy=noisy, screenshot=False, k = self.k)

        # self.s_t = None

        # Discrete action space
        # self.steers = [-0.50, 0, 0.50]
        # self.action_space = gym.spaces.Discrete(len(self.steers))

    def reset(self, track_offset=0):
        relaunch = False

        if self.episode % 3 == 0:
            relaunch = True

        self.episode += 1
        self.step_per_episode = 0

        self.last_steer = 0.0

        ob = self.env.reset(relaunch=relaunch, track_offset=track_offset)
        # print(ob.img)
        # cv2.resize(ob.img, (320, 240))[:, 40:280]
        # print(np.asarray(ob.img).shape)
        # img = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 127.5 - 1
        # img = img.reshape(64, 64, 1)
        # self.s_t = np.stack((img, img), axis=2)
        # self.dist_start = ob.distFromStart
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, self.last_steer))

        return s_t

    def step(self, action):
        self.step_total += 1
        self.step_per_episode += 1

        if self.control_dim == 1:
            control = [action[0], self.throttle, 0]
        else :
            control = action

        ob, reward, done, _ = self.env.step(control)

        # img = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 127.5 - 1
        # img = img.reshape(img.shape[0], img.shape[1], 1)
        # print(self.s_t.shape)
        # self.s_t = np.append(self.s_t[:, :, 1:], img, axis=2)

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, self.last_steer))

        self.last_steer = action[0]
        return s_t, reward, done, None # ob.distFromStart - self.dist_start

    def end(self):
        self.env.end()

if __name__ == "__main__":
    env = TorcsWrapper(port=11111, control_dim=3, throttle=0.20)
    ob = env.reset(0)
    while True:
        # cv2.imshow("img", img[0][:, :, 1])
        # cv2.waitKey(1)
        # print(img[0].shape)
        # print(img[1])
        # print(img)
        ob, _, done, info = env.step([0, 0.2, 0])
        print(ob.shape)
        if done == True:
            break
    env.end()
