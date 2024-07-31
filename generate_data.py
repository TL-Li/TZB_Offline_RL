import json
import copy
import random


def generate_info(seeds):
    random.seed(seeds)
    height = 10000
    p1 = [round(random.uniform(118.0, 119.0), 2), round(random.uniform(25.0, 26.0), 2), height]
    p2 = [round(random.uniform(118.0, 119.0), 2), round(random.uniform(25.0, 26.0), 2), height]
    p3 = [round(random.uniform(118.0, 119.0), 2), round(random.uniform(25.0, 26.0), 2), height]
    p4 = [round(random.uniform(118.0, 119.0), 2), round(random.uniform(25.0, 26.0), 2), height]
    kill = random.randint(0, 4)
    death = random.randint(0, 4)

    return p1, p2, p3, p4, kill, death
    

# 创建一个空的字典，用于存储数据
data_dict = {}
info = {}

# 添加每条数据到字典中
for i in range(1, 1101):
    p1, p2, p3, p4, k, d = generate_info(i)
    data_dict["100000"] = p1
    data_dict["200000"] = p2
    data_dict["300000"] = p3
    data_dict["400000"] = p4
    
    data_dict["kill"] = k
    data_dict["death"] = d
    info[i] = copy.deepcopy(data_dict) # 避免地址映射


# 将数据写入 JSON 文件
with open('data.json', 'w') as json_file:
    json.dump(info, json_file, indent=4)

print("数据已写入文件 data.json")

    