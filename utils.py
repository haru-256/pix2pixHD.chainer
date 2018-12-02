import pathlib
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.backends.cuda
from chainer import Variable
from chainer.backends import cuda


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(np.sqrt(total))
    rows = int(np.ceil(float(total) / cols))
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros(
        (height * rows, width * cols, 3), dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        for ch in range(3):
            combined_image[width*i:width*(i+1), height*j:height*(j+1), ch] =\
                image[:, :, ch]
    return combined_image


def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        n_images = rows * cols

        xp = gen.xp  # get module
        xp.random.seed(seed)  # fix seed
        np.random.seed(seed)  # fix seed
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                x = gen(z)
        x = chainer.backends.cuda.to_cpu(x.data)
        np.random.seed()

        x = (x * 127.5 + 127.5) / 255  # 0~255に戻し0~1へ変形
        x = x.transpose(0, 2, 3, 1)  # NCHW->NHWCに変形
        x = combine_images(x)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
        axes.imshow(x)
        axes.axis("off")
        preview_dir = pathlib.Path('{}/preview'.format(dst))
        preview_path = preview_dir /\
            'image_{:}epoch.jpg'.format(trainer.updater.epoch)
        if not preview_dir.exists():
            preview_dir.mkdir()
        axes.set_title("epoch: {}".format(trainer.updater.epoch), fontsize=18)
        fig.tight_layout()
        fig.savefig(preview_path)
        plt.close(fig)

    return make_image


class Gamma_initializer(chainer.initializer.Initializer):
    """
    Return Normal initializer whose mean is "mean", std is "scale".

    Parameters
    ---------------

    mean: float
        mean of Normal distribution.

    scale: float
        standard deviation of Normal distribution.

    dtype: Data type specifier.
    """

    def __init__(self, mean, scale, dtype=None):
        self.mean = mean
        self.scale = scale
        super(Gamma_initializer, self).__init__(dtype=None)

    def __call__(self, array):
        """
        Initializes given array.

        Parameters
        ------------

        array: numpy.ndarray or cupy.ndarray
            An array to be initialized by this initializer.
        """
        xp = cuda.get_array_module(array)
        args = {'loc': self.mean, 'scale': self.scale, 'size': array.shape}
        if xp is not np:
            # Only CuPy supports dtype option
            if self.dtype == np.float32 or self.dtype == np.float16:
                # float16 is not supported in cuRAND
                args['dtype'] = np.float32
        array[...] = xp.random.normal(**args)
