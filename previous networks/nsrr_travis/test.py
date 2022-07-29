from factories import LossMaker
import cv2


bilin = cv2.imread('./results/run_4/val_bilin.png')
res = cv2.imread('./results/run_4/val_restored.png')

loss = LossMaker()
loss.total_loss()

