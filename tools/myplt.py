import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import json
data_path_1 = "D:/download/Code/mmsegmentation-0.24.1/mmseg_log/a/oursstdc/stdc0-20221109_104704.log.json"
data_path_2 = "D:/download/Code/mmsegmentation-0.24.1/mmseg_log/a/oursstdc/bisenetv1-20230412_150908.log.json"
data_path_3 = "D:/download/Code/mmsegmentation-0.24.1/mmseg_log/a/oursstdc/bisnetv2-5.2-20230502_225051.log.json"
data_path_4 = "D:/download/Code/mmsegmentation-0.24.1/mmseg_log/a/oursstdc/deeplabv3_r18_20230512_174855.log.json"
data_path_5 = "D:/download/Code/mmsegmentation-0.24.1/mmseg_log/a/oursstdc/deeplabv3plus-20230512_174654.log.json"
data_path_6 = "D:/download/Code/mmsegmentation-0.24.1/mmseg_log/a/oursstdc/ours-20230412_183114.log.json"
def read_json(data_path, stage, key_1, key_2):
    iter, acc_seg = [], []
    with open(data_path, 'r') as f:
        content = f.readlines()
        for line in content:
            if json.loads(line)["mode"] == stage:
                iter.append(json.loads(line)[key_1]*100/350)
                acc_seg.append(json.loads(line)[key_2])
    return iter, acc_seg
def read_json2(data_path, stage, key_1, key_2):
    iter, acc_seg = [], []
    with open(data_path, 'r') as f:
        content = f.readlines()
        for line in content:
            if json.loads(line)["mode"] == stage:
                iter.append(json.loads(line)[key_1])
                acc_seg.append(json.loads(line)[key_2]-15)
    return iter, acc_seg
# iter_1, acc_seg_1 = read_json(data_path_1, "train", "iter", "decode.acc_seg")
# iter_2, acc_seg_2 = read_json(data_path_2, "train", "iter", "decode.acc_seg")
# # iter_3, acc_seg_3 = read_json(data_path_3, "train", "iter", "decode.acc_seg")
# iter_4, acc_seg_4 = read_json(data_path_4, "train", "iter", "decode.acc_seg")
# iter_5, acc_seg_5 = read_json(data_path_5, "train", "iter", "decode.acc_seg")
# iter_6, acc_seg_6 = read_json(data_path_6, "train", "iter", "decode.acc_seg")
iter_1, acc_seg_1 = read_json(data_path_1, "val", "epoch", "IoU.sidewalk")
iter_2, acc_seg_2 = read_json(data_path_2, "val", "epoch", "IoU.sidewalk")
iter_3, acc_seg_3 = read_json(data_path_3, "val", "epoch", "IoU.sidewalk")
iter_4, acc_seg_4 = read_json(data_path_4, "val", "epoch", "IoU.sidewalk")
iter_5, acc_seg_5 = read_json(data_path_5, "val", "epoch", "IoU.sidewalk")
iter_6, acc_seg_6 = read_json(data_path_6, "val", "epoch", "IoU.sidewalk")
max_value = max(acc_seg_1) if max(acc_seg_1) > max(acc_seg_2) else max(acc_seg_2)
print(max_value)
#设置横纵坐标的名称以及对应字体格式
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 10,
}
index = 0
# for i in range(len(acc_seg_2)):
#     if acc_seg_2[i] >= 80:
#         index = i
#         break
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xlim(10,100)
plt.ylim(0.44,0.66)
ax.set_xlabel('epoch', font)
ax.set_ylabel('IoU', font)
ax.set_title('IoU',size=10)
x_major_locator=MultipleLocator(10)
y_major_locator=MultipleLocator(0.01)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.plot(iter_1[index:], acc_seg_1[index:], label = "STDC")
ax.plot(iter_2[index:], acc_seg_2[index:], label = "BiseNetV1")
# ax.plot(iter_3[index:], acc_seg_3[index:], label = "BiseNetV2")
ax.plot(iter_4[index:], acc_seg_4[index:], label = "Deeplabv3")
ax.plot(iter_5[index:], acc_seg_5[index:], label = "Deeplabv3+")
ax.plot(iter_6[index:], acc_seg_6[index:], label = "ours")
plt.legend(loc='lower right')
plt.show()
