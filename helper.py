import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores, image_file_name=None, win: bool = False):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    if win:
        plt.showfig(image_file_name)
    plt.show(block=False)
    plt.pause(.1)


if __name__ == '__main__':
    plot([0], [0.0])
    plot([0, 0], [0.0, 0.0])
    plot([0, 0, 1], [0.0, 0.0, 0.333])
    plot([0, 0, 1, 0], [0.0, 0.0, 0.333, 0.25])

