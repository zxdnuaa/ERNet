import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import json
data_path_1 = r"D:\download\Code\mmsegmentation-0.24.1\mmseg_log\a\deepcrack\bisenetv1\20230526_231309.log.json"
data_path_2 = r"D:\download\Code\mmsegmentation-0.24.1\mmseg_log\a\deepcrack\deeplabv3plus\20230526_231516.log.json"
data_path_3 = r"D:\download\Code\mmsegmentation-0.24.1\mmseg_log\a\deepcrack\stdc2\20230526_233141.log.json"
data_path_4 = r"D:\download\Code\mmsegmentation-0.24.1\mmseg_log\a\deepcrack\ours_ep=100\20230527_205402.log.json"
# data_path_5 = "D:/download/Code/mmsegmentation-0.24.1/mmseg_log/a/oursstdc/deeplabv3plus-20230512_174654.log.json"
# data_path_6 = "D:/download/Code/mmsegmentation-0.24.1/mmseg_log/a/oursstdc/ours-20230412_183114.log.json"
def read_json(data_path, stage, key_1, key_2):
    iter, acc_seg = [], []
    with open(data_path, 'r') as f:
        content = f.readlines()
        for line in content:
            if json.loads(line)["mode"] == stage:
                iter.append(json.loads(line)[key_1]*100/3750)
                acc_seg.append(json.loads(line)[key_2])
    return iter, acc_seg
def read_json2(data_path, stage, key_1, key_2):
    iter, acc_seg = [], []
    with open(data_path, 'r') as f:
        content = f.readlines()
        for line in content:
            if json.loads(line)["mode"] == stage:
                iter.append(json.loads(line)[key_1]*100/352)
                acc_seg.append(json.loads(line)[key_2])
    return iter, acc_seg
iter_1, acc_seg_1 = read_json(data_path_1, "train", "iter", "decode.acc_seg")
iter_2, acc_seg_2 = read_json(data_path_2, "train", "iter", "decode.acc_seg")
iter_3, acc_seg_3 = read_json(data_path_3, "train", "iter", "decode.acc_seg")
iter_4, acc_seg_4 = read_json(data_path_4, "train", "iter", "decode.acc_seg")
# iter_5, acc_seg_5 = read_json(data_path_5, "train", "iter", "decode.acc_seg")
# iter_6, acc_seg_6 = read_json(data_path_6, "train", "iter", "decode.acc_seg")

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
# plt.xlim(-5,100)
# plt.ylim(0.44,0.66)
ax.set_xlabel('epoch', font)
ax.set_ylabel('acc_seg', font)
ax.set_title('Deep-Crack Seg-acc',size=10)
x_major_locator=MultipleLocator(10)
# y_major_locator=MultipleLocator(0.01)
ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
ax.plot(iter_1[index:], acc_seg_1[index:], label = "BiseNetV1")
ax.plot(iter_2[index:], acc_seg_2[index:], label = "deeplabv3+")
ax.plot(iter_3[index:], acc_seg_3[index:], label = "STDC")
ax.plot(iter_4[index:], acc_seg_4[index:], label = "ours")
plt.legend(loc='lower right')
plt.show()
