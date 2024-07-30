import json
import numpy as np
from gym import spaces
def load_data(file_path="./data.json"):
	
	with open(file_path, 'r', encoding='utf-8') as file:  
    	# 使用json.load()方法解析JSON数据
		data = json.load(file)
	# first = data["1"]
	# l1 = first["100000"]
	# print(first)
	# print(l1)
	dataset = {
		"observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
		"terminals": []
	}
	next_obs = np.concatenate((data["1"]["100000"], data["1"]["200000"], data["1"]["300000"], data["1"]["400000"]))
	dataset['observations'] = np.zeros(12)
	dataset['actions'] = next_obs
	dataset['next_observations'] = next_obs
	dataset['rewards'] = data["1"]["kill"] - data["1"]["death"]
	dataset['terminals'] = 1
	print(dataset)


if __name__ == "__main__":

	state = np.zeros(12)#状态空间
	observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12, ), dtype=np.float64)
	print(observation_space)
        
	# load_data()