import json
import copy
import random


def generate_info(seeds):
    random.seed(seeds)
    height = 10000
    p1 = [round(random.uniform(118.45, 118.55), 2), round(random.uniform(25.15, 25.25), 2), height]
    p2 = [round(random.uniform(118.65, 118.75), 2), round(random.uniform(24.95, 25.05), 2), height]
    p3 = [round(random.uniform(118.85, 118.95), 2), round(random.uniform(25.15, 25.25), 2), height]
    p4 = [round(random.uniform(119.05, 119.15), 2), round(random.uniform(24.95, 25.05), 2), height]
    kill = random.randint(0, 3)
    death = random.randint(0, 4)

    return p1, p2, p3, p4, kill, death

def generate_info_good():
    height = 10000
    p1 = [118.5, 25.2, height]
    p2 = [118.7, 25.0, height]
    p3 = [118.9, 25.2, height]
    p4 = [119.1, 25.0, height]
    kill = 4
    death = 0

    return p1, p2, p3, p4, kill, death
    

# 创建一个空的字典，用于存储数据
data_dict = {}
info = {}

# 添加每条数据到字典中
for i in range(1, 1500):
    if i % 17 == 0:
        p1, p2, p3, p4, k, d = generate_info_good()
    else:
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

    