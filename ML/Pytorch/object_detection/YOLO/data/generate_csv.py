import os
import csv

read_train = open("train.txt", "r").readlines()

with open("train.csv", mode="w", newline="") as train_file:
    for line in read_train:
        image_file = line.split("/")[-1].replace("\n", "")
        text_file = image_file.replace(".jpg", ".txt")
        data = [image_file, text_file]
        writer = csv.writer(train_file)
        writer.writerow(data)

read_train = open("test.txt", "r").readlines()

with open("test.csv", mode="w", newline="") as train_file:
    for line in read_train:
        image_file = line.split("/")[-1].replace("\n", "")
        text_file = image_file.replace(".jpg", ".txt")
        data = [image_file, text_file]
        writer = csv.writer(train_file)
        writer.writerow(data)
