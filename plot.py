import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_score):
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores, label = 'Score')
    plt.plot(mean_score, label = 'Mean Score')
    plt.legend()
    plt.ylim(ymin = min(scores)-10 if scores else 0)
    plt.pause(0.1)