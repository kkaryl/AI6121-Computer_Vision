from __future__ import print_function
import matplotlib.pyplot as plt

__all__ = ['plot_images']

def plot_images(images, labels, preds=None):
    # https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-utils-py
    assert len(images) == len(labels) == 9
    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0, :, :], interpolation='spline16', cmap='gray')
            
        label = str(labels[i])
        if preds is None:
            xlabel = label
        else:
            pred = str(preds[i])
            xlabel = "True: {0}\nPred: {1}".format(label, pred)
            
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    https://github.com/amitrajitbose/
    '''
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()