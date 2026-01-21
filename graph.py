"""
可视化工具 (Visualization)

绘制训练过程中的指标曲线：
- 训练损失和验证损失曲线
- BLEU 分数曲线
用于分析模型训练效果和收敛情况
"""

import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(mode):
    if mode == 'loss':
        # train = read('./result/train_loss.txt')
        # test = read('./result/test_loss.txt')
        train = read('./saved/transformer-base/train.txt')
        test = read('./saved/transformer-base/test.txt')
        plt.plot(train, 'r', label='train')
        plt.plot(test, 'b', label='validation')
        plt.legend(loc='lower left')


    elif mode == 'bleu':
        bleu = read('./result/bleu.txt')
        plt.plot(bleu, 'b', label='bleu score')
        plt.legend(loc='lower right')

    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.show()


if __name__ == '__main__':
    draw(mode='loss')
    draw(mode='bleu')
