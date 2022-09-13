import json
import os
import csv

f = open('label.csv','w',newline='')
wr = csv.writer(f)

for filename in os.listdir("D:/data/json"):
    with open(os.path.join("D:/data/json", filename), 'r', encoding='UTF-8') as file:
        data = json.load(file)
        arr = [data["metaData"]["duration"]]
        for i in range(len(data["data"])):
           arr.append(data["data"][i]["attributes"][0]["name"])
        wr.writerow(arr)
f.close()