import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

DATA_GRAPH_PATH = './data_graph'


def save_fig_schedule_accuracy(fig_path, data, type, name, score=0.0):
    plt.figure(figsize=(10, 5))
    plt.plot(data, 'r--')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Figure {} {}. Score : {:.2f}'.format(type, name, score))
    plt.savefig('{}/{}_{}.png'.format(fig_path, type, name), dpi=100)
    # print('Saved Figure')


def save_fig_schedule_loss(fig_path, data, type, name):
    plt.figure(figsize=(10, 5))
    plt.plot(data, 'r--')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Figure Loss {}'.format(type))
    plt.savefig('{}/{}_{}.png'.format(fig_path, type, name), dpi=100)
    # print('Saved Figure')


def save_fig_label_data(path, name):
    array = np.zeros(10, dtype=np.int)
    files = Path(path).glob('*.wav')
    for file in files:
        label = file.name.split('_')[0]
        array[int(label)] += 1
    left = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    height_bar = list(array)

    # labels for bars
    tick_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    rect = ax.bar(left, height_bar, tick_label=tick_label, width=0.8, color='blue')
    plt.title("Total data: {}".format(array.sum()))
    plt.xlabel('Label')
    plt.ylabel('Total')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rect)
    plt.gcf().savefig('{}/{}.png'.format(DATA_GRAPH_PATH, name), dpi=100)


def save_confusion_matrix(matrix, name, dst_path):
    alpha = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.GnBu)

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(alpha)
    ax.set_yticklabels(alpha)
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - .5, minor=True)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            c = matrix[i, j]
            ax.text(i, j, str(c), va='center', ha='center')
    plt.gcf().savefig('{}/{}.png'.format(dst_path, name), dpi=60)


if __name__ == '__main__':
    # a = np.random.randint(0, 100, size=(10, 10))
    # save_confusion_matrix(a)
    # if not os.path.isdir(DATA_GRAPH_PATH):
    #     os.mkdir(DATA_GRAPH_PATH)
    # save_fig_label_data('./test', 'sdata_test')
    # save_fig_label_data('./data_to_train', 'data_to_train')
    # save_fig_label_data('./data', 'data')
    # save_fig_label_data('./val', 'val')
    # save_fig_label_data('./train', 'train')
    accuracy = np.zeros(10)
    score = np.zeros(10)
    models_file = Path('./model').glob('*.pth')
    for idx, model in enumerate(models_file):
        state = torch.load(model, map_location=torch.device('cpu'))
        print('{} : {:.2f}'.format(model.name, state['best_acc']*100))