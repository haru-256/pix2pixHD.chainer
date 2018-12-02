import chainer
from chainer import Variable
import chainer.functions as F


class DCGANUpdater(chainer.training.updaters.StandardUpdater):
    """
    costomized updater for DCGAN

    Upedater の自作は，基本的に__init__とupdate_core overrideすればよい
    update_core は1バッチの更新処理を書く
    """

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop("models")  # extract model

        super(DCGANUpdater, self).__init__(*args,
                                           **kwargs)  # StandardUpdaterを呼ぶ

    def update_core(self):
        # get_optimizer mehtod allows to get optimizer
        gen_optimizer = self.get_optimizer("gen")
        dis_optimizer = self.get_optimizer("dis")
        gen, dis = self.gen, self.dis

        # obtain batch data
        # get_iterator("main") is SerialIterator so next() returns next minibatch
        batch = self.get_iterator("main").next()

        x_real = self.converter(
            batch, self.device
        )  # self.converter() is concat_example() また self.deviceでデータをgpuに送る
        x_real = Variable(x_real)
        x_real = (x_real - 127.5) / 127.5  # normalize image data
        xp = chainer.backends.cuda.get_array_module(
            x_real.data)  # return cupy or numpy based on type of x_real.data
        batch_size = len(batch)

        # inference
        y_real = dis(x_real)  # Genuine image estimation result

        z = Variable(xp.asarray(gen.make_hidden(
            batch_size)))  # genertate z random vector  xp.asarrayでcupy形式に変更する
        x_fake = gen(z)  # genertate fake data by generator

        y_fake = dis(x_fake)  # Estimation result of fake image

        # optimize dis, gen respectively
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)  # dis/loss でアクセス可能
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)  # gen/loss でアクセス可能
        return loss


class WGANUpdater(chainer.training.updaters.StandardUpdater):
    """
    costomized updater for WGAN

    Paramerers
    --------------
    iterator 
        – Dataset iterator for the training dataset. 
        It can also be a dictionary that maps strings to iterators. 
        If this is just an iterator, then the iterator is registered by the name 'main'.

    optimizer 
        – Optimizer to update parameters.
        It can also be a dictionary that maps strings to optimizers.
        If this is just an optimizer, then the optimizer is registered by the name 'main'.

    converter 
        – Converter function to build input arrays.
        Each batch extracted by the main iterator and the device option
        are passed to this function. concat_examples() is used by default.

    device 
        – Device to which the training data is sent.
        Negative value indicates the host memory (CPU).

    loss_func 
        – Loss function.
        The target link of the main optimizer is used by default.

    loss_scale (float) 
        – Loss scaling factor. Loss scaling is a usefull technique
        to mitigate vanishing gradient issue that tends to happen
        when low precision data type like float16 is used during training.
        If you set loss scaling factor, gradients of loss values are to be multiplied by
        the factor before backprop starts. The factor is propagated to whole gradients
        in a computational graph along the backprop. The gradients of parameters are divided
        by the factor just before the parameters are to be updated.
    """

    def __init__(self, *args, **kwargs):
        self.gen, self.critic = kwargs.pop("models")  # extract model
        self.n_critic = kwargs.pop("n_critic")  # extract n_critic
        super(WGANUpdater, self).__init__(*args,
                                          **kwargs)  # StandardUpdaterを呼ぶ

    def update_core(self):
        # get_optimizer mehtod allows to get optimizer
        gen_optimizer = self.get_optimizer("gen")
        critic_optimizer = self.get_optimizer("critic")

        # obtain batch data
        # get_iterator("main") is SerialIterator so next() returns next minibatch
        batch = self.get_iterator("main").next()

        # optimize Critic
        for n in range(self.n_critic):
            x_real = self.converter(
                batch, self.device
            )  # self.converter() is concat_example() また self.deviceでデータをgpuに送る
            x_real = Variable(x_real)
            x_real = (x_real - 127.5) / 127.5  # normalize image data
            xp = chainer.backends.cuda.get_array_module(
                x_real.data)  # return cupy or numpy based on type of x_real.data
            batch_size = len(batch)

            # inference
            y_real = self.critic(x_real)  # Genuine image estimation result

        z = Variable(xp.asarray(self.gen.make_hidden(
            batch_size)))  # genertate z random vector  xp.asarrayでcupy形式に変更する
        x_fake = self.gen(z)  # genertate fake data by generator

        y_fake = self.critic(x_fake)  # Estimation result of fake image

        # optimize dis, gen respectively
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)  # dis/loss でアクセス可能
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)  # gen/loss でアクセス可能
        return loss
