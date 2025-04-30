import numpy as np
import matplotlib.pyplot as plt

scores = np.load('scores.npy')

plt.figure()
plt.title('Training Scores')
scores_clipped = np.clip(scores, -1000, 100)
plt.plot(scores_clipped, label = 'scores')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.ylim(ymin =min(scores)-5)
plt.show
print(scores[-5:])