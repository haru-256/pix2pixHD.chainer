import random
import chainer
from chainer import training
from chainer.training import extensions
from chainer.datasets import ImageDataset
from chainer.serializers import save_npz
from generator64 import Generator
from updater import DCGANUpdater
from visualize import out_generated_image
# from accuracy_reporter import accuracy_report
import pathlib
import matplotlib.pyplot as plt
import pathlib


def make_optimizer(model, alpha=0.0002, beta1=0.5):
    """
    Setup an optimizer
    """
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0001), 'hook_dec')

    return optimizer


if __name__ == '__main__':
    import numpy as np
    import argparse

    # パーサーを作る
    parser = argparse.ArgumentParser(
        prog='train',  # プログラム名
        usage='train DCGAN',  # プログラムの利用方法
        description='description',  # 引数のヘルプの前に表示
        epilog='end',  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )

    # 引数の追加
    parser.add_argument('-s', '--seed', help='seed',
                        type=int, required=True)
    parser.add_argument('-n', '--number', help='the number of experiments.',
                        type=int, required=True)
    parser.add_argument('--hidden', help='the number of codes of Generator.',
                        type=int, default=100)
    parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 300',
                        type=int, default=300)
    parser.add_argument('-bs', '--batch_size', help='batch size. defalut value is 128',
                        type=int, default=128)
    parser.add_argument('-g', '--gpu', help='specify gpu by this number. defalut value is 0',
                        choices=[0, 1], type=int, default=0)
    parser.add_argument('-ks', '--ksize',
                        help='specify ksize of generator by this number. any of following;'
                        ' 4 or 6. defalut value is 4',
                        choices=[4, 6], type=int, default=4)
    parser.add_argument('-dis', '--discriminator',
                        help='specify discriminator by this number. any of following;'
                        ' 0: original, 1: minibatch discriminatio, 2: feature matching, 3: GAP. defalut value is 0',
                        choices=[0, 1, 2, 3], type=int, default=0)
    parser.add_argument('-ts', '--tensor_shape',
                        help='specify Tensor shape by this numbers. first args denotes to B, seconds to C.'
                        ' defalut value are B:32, C:8',
                        type=int, default=[32, 8], nargs=2)
    parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                        action='version',
                        default=False)

    # parse arguments
    opt = parser.parse_args()
    out = pathlib.Path(
        "result_{1}/result_{1}_{2}".format(opt.number, opt.seed)).resolve()

    # set seed
    # use cuDNN so dose not determinstic
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    # set Chainer(CuPy) random seed
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(opt.seed)
        if chainer.backends.cuda.cudnn_enabled:
            print("# Cant't assue determinstic")

    # make directory
    cdir = pathlib.Path('.').resolve()
    for path in out.parents:
        if not path.exists():
            path.mkdir()

    # write arguments to file:arg.txt
    with open(out / "args.txt", "w") as f:
        f.write(repr(opt))
    print('# Arguments:', opt)

    # Set up a generator
    gen = Generator(n_hidden=n_hidden, ksize=args.ksize, pad=pad)

    if gpu >= 0:
        # specific GPU current
        chainer.backends.cuda.get_device_from_id(opt.gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # Prepare Dataset
    data_dir = pathlib.Path("../data/CelebA/Img/center_cropped_resize_64/")
    abs_data_dir = data_dir.resolve()
    print("# data dir path:", abs_data_dir)
    data_path = [path for path in abs_data_dir.glob("*.jpg")]
    print("# data length:", len(data_path))
    data = ImageDataset(paths=data_path)  # dtype=np.float32

    # Prepare Iterator
    train_iter = chainer.iterators.SerialIterator(data, opt.batch_size)

    # Set up a updater and trainer
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen,
            'dis': opt_dis
        },
        device=opt.gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    snapshot_interval = (1, 'epoch')
    display_interval = (1, 'epoch')
    # storage method is hdf5
    trainer.extend(
        extensions.snapshot(
            filename='snapshot_epoch_{.updater.epoch}.npz',
            savefun=save_npz),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(
            gen, 'gen_epoch_{.updater.epoch}.npz', savefun=save_npz),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(
            dis, 'dis_epoch_{.updater.epoch}.npz', savefun=save_npz),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport([
            'epoch', 'iteration', 'gen/loss', 'dis/loss', 'elapsed_time',
        ]),
        trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=20))
    trainer.extend(
        out_generated_image(gen, dis, 5, 5, opt.seed, out),
        trigger=display_interval)
    trainer.extend(
        extensions.PlotReport(
            ['gen/loss', 'dis/loss'],
            x_key='epoch',
            file_name='loss_{0}_{1}.jpg'.format(opt.number, opt.seed),
            grid=False))
    trainer.extend(extensions.dump_graph("gen/loss", out_name="gen.dot"))
    trainer.extend(extensions.dump_graph("dis/loss", out_name="dis.dot"))

    # Run the training
    trainer.run()
