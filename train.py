from utils import step_lr_scheduler, Averagvalue, save_checkpoint
from utils import cross_entropy_loss, uncertainty_loss, ELBO_loss
import time
import torchvision
from os.path import join
import os
import torch
from dataset.transformer import cutmix
from torch.distributions import Normal, Independent


def train(train_loader, model, optimizer, epoch, args, logger=None):
    optimizer = step_lr_scheduler(optimizer, epoch, args.stepsize)
    save_dir = join(args.savedir, 'epoch-%d-training-record' % epoch)
    os.makedirs(save_dir, exist_ok=True)

    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()

    # switch to train mode
    model.train()
    if logger is not None:
        info = "Epoch: {}  Pret lr: {}  Unpret lr: {}".format(
            epoch, optimizer.state_dict()['param_groups'][0]['lr'],
            optimizer.state_dict()['param_groups'][2]['lr'])
        logger.info(info)
    elif args.rank == 0:
        print(epoch,
              "Pretrained lr:", optimizer.state_dict()['param_groups'][0]['lr'],
              "Unpretrained lr:", optimizer.state_dict()['param_groups'][2]['lr'])

    end = time.time()
    epoch_loss = []
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()
        outputs = model(image)
        counter += 1
        loss = cross_entropy_loss(outputs, label, args.loss_lmbda)
        loss = loss / args.itersize
        loss.backward()

        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0

        losses.update(loss, image.size(0))
        epoch_loss.append(loss)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0 and args.rank == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.2f} (avg:{batch_time.avg:.2f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=losses)
            if logger is not None:
                logger.info(info)
            else:
                print(info)
            label[label == 2] = 0.5
            if args.batch_size > 4:
                outputs, label = outputs[:4, ...], label[:4, ...]

            outputs = torch.cat([outputs, label], dim=0)
            torchvision.utils.save_image(outputs, join(save_dir, "iter-%d.jpg" % i),
                                         nrow=outputs.size(0) // 2)

    if args.rank == 0:
        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))


def semi_train(ldataloader, udataloader, model, nms_model, optimizer, epoch, args, cfg, logger):
    optimizer = step_lr_scheduler(optimizer, epoch, args.stepsize)
    save_dir = join(args.savedir, 'epoch-%d-training-record' % epoch)
    os.makedirs(save_dir, exist_ok=True)

    if cfg["model"]["loss"] == "WCE":
        from utils import cross_entropy_loss
        loss_func = cross_entropy_loss
    elif cfg["model"]["loss"] == "UNCERT":
        from utils import uncertainty_loss
        loss_func = uncertainty_loss
    elif cfg["model"]["loss"] == "ELBO":  # (Evidence Lower BOund)
        from utils import ELBO_loss
        loss_func = ELBO_loss
    else:
        raise Exception("loss function error")

    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    llosses = Averagvalue()
    ulosses = Averagvalue()

    # switch to train mode
    model.train()
    if logger is not None:
        info = "Epoch: {}  Pret lr: {}  Unpret lr: {}".format(
            epoch, optimizer.state_dict()['param_groups'][0]['lr'],
            optimizer.state_dict()['param_groups'][2]['lr'])
        logger.info(info)

    assert len(ldataloader) == len(udataloader)
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, ((image, label), (wimg, simg)) in enumerate(zip(ldataloader, udataloader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if cfg["dataset"]["CutMix"]:
            # bug 将wimg写成wing，但是wimg在函数里面执行的是原地操作，因此实际上函数外面的wimg也cutmix了
            # 真是 一个bug是bug，两个bug能work 啊
            # wing, simg = cutmix(wimg, simg, p=0.5)
            wimg, simg = cutmix(wimg, simg, p=0.5)
            # 不应该对wimg进行cutmix，应该对预测的结果cutmix，重新训一下看看

        image, label, wimg, simg = image.cuda(), label.cuda(), wimg.cuda(), simg.cuda()

        outputs = model(torch.cat((image, simg)))

        if cfg["model"]["loss"] == "ELBO":
            mean, std = outputs
            outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
            outputs = torch.sigmoid(outputs_dist.rsample())

            ol, ou = torch.chunk(outputs, chunks=2)
            ml, mu = torch.chunk(mean, chunks=2)
            sl, su = torch.chunk(std, chunks=2)

            lloss = loss_func(ol, label, ml, sl, args.loss_lmbda)


            with torch.no_grad():
                m, s = model(wimg)
                wlb = nms_model(torch.sigmoid(m + s))
            uloss = loss_func(ou, wlb, mu, su, args.loss_lmbda)

        else:
            ol, ou = torch.chunk(outputs, chunks=2)
            lloss = loss_func(ol, label, args.loss_lmbda)

            with torch.no_grad():
                wlb = model(wimg)
                wlb = nms_model(wlb)
            uloss = loss_func(ou, wlb, args.loss_lmbda)

        # loss = lloss + int(epoch != 0) * args.loss_hype * uloss
        loss = lloss + args.loss_hype * uloss

        loss = loss / args.itersize
        loss.backward()

        counter += 1

        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0

        losses.update(loss, image.size(0))
        llosses.update(lloss, image.size(0))
        ulosses.update(uloss, image.size(0))
        epoch_loss.append(loss)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % (len(ldataloader) // args.print_freq) == 0 and logger is not None:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(ldataloader)) + \
                   'Time {batch_time.val:.2f} (avg:{batch_time.avg:.2f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=losses) + \
                   'LLoss {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=llosses) + \
                   'ULoss {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=ulosses)
            logger.info(info)

            wlb[wlb == 2] = 0.5
            label[label == 2] = 0.5

            if cfg["model"]["loss"] == "ELBO":
                if args.batch_size > 4:
                    image = image[:4, ...]
                    ml = ml[:4, ...]
                    sl = sl[:4, ...]
                    ol = ol[:4, ...]
                    label = label[:4, ...]
                    simg = simg[:4, ...]
                    mu = mu[:4, ...]
                    su = su[:4, ...]
                    ou = ou[:4, ...]
                    wlb = wlb[:4, ...]

                outputs = torch.cat([image,  # images
                                     ml.repeat(1, 3, 1, 1),  # mean(labeled)
                                     sl.repeat(1, 3, 1, 1),  # std(labeled)
                                     ol.repeat(1, 3, 1, 1),  # output(labeled)
                                     label.repeat(1, 3, 1, 1),  # label
                                     simg,  # strong enhence images
                                     mu.repeat(1, 3, 1, 1),  # mean (unlabeled)
                                     su.repeat(1, 3, 1, 1),  # std (unlabeled)
                                     ou.repeat(1, 3, 1, 1),  # # output(unlabeled)
                                     wlb.repeat(1, 3, 1, 1)], dim=0)  # weak enhence images as label
                torchvision.utils.save_image(outputs, join(save_dir, "iter-%d.jpg" % i),
                                             nrow=min(4, args.batch_size))
            else:
                if args.batch_size > 4:
                    image = image[:4, ...]
                    ol = ol[:4, ...]
                    label = label[:4, ...]
                    simg = simg[:4, ...]
                    ou = ou[:4, ...]
                    wlb = wlb[:4, ...]
                outputs = torch.cat([image,
                                     ol.repeat(1, 3, 1, 1),
                                     label.repeat(1, 3, 1, 1),
                                     simg,
                                     ou.repeat(1, 3, 1, 1),
                                     wlb.repeat(1, 3, 1, 1)], dim=0)

                torchvision.utils.save_image(outputs, join(save_dir, "iter-%d.jpg" % i),
                                             nrow=min(4, args.batch_size))

    # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))
