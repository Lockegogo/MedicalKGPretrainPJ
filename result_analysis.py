import matplotlib.pyplot as plt
import numpy as np


def plot(train_loss, val_auc):
    fig, ax1 = plt.subplots()
    ax1.plot(
        np.arange(len(train_loss)), np.array(train_loss), c="blue", label="train loss"
    )
    ax1.set_xlabel("train epochs", fontsize=13)
    ax1.set_ylabel("train loss", fontsize=13)
    ax1.set_ylim([0, 1])

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(val_auc)), np.array(val_auc), c="red", label="val auc")
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("auc in val set", fontsize=13)
    ax2.set_ylim([0, 1])
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig("results/train_val.png")



def analysis():
    pass

