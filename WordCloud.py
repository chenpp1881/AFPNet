import wordcloud
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cloud_txt_path',type=str)
parser.add_argument('--cloud_max_words',type=int,default=30)
opts = parser.parse_args()

def gen_wordcloud(opts):
    with open(opts.cloud_txt_path) as f:
        text = f.read()

    # 配置词云参数
    wc = wordcloud.WordCloud(background_color='white', max_words=opts.cloud_max_words, contour_width=3, scale=4,
                             contour_color='steelblue')

    # 生成词云图像
    wc.generate(text)

    # 显示词云
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    gen_wordcloud(opts)