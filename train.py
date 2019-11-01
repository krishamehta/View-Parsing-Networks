from utils import Foo
from models import VPNModel
from datasets import Matterport_Dataset
from opts import parser
from transform import *
import torchvision
import torch
from torch import nn
from torch import optim
import os
import time
import shutil
import random

mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]

points_filepath = 'pairs.txt'
points = []
with open(points_filepath) as f:
    for line in f:
        points.append(line[:-1])

random.shuffle(points)

def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    network_config = Foo(
        encoder=args.encoder,
        decoder=args.decoder,
        fc_dim=args.fc_dim,
        output_size=args.label_resolution,
        num_views=args.n_views,
        num_class=args.num_class,
        transform_type=args.transform_type,
    )

    split_size = int(0.8*len(points))
    train_list = points[:split_size]
    val_list = points[split_size:]

    train_dataset = Matterport_Dataset(args.data_root, train_list,
                         transform=torchvision.transforms.Compose([
                             Stack(roll=True),
                             ToTorchFormatTensor(div=True),
                             GroupNormalize(mean_rgb, std_rgb)
                         ]),
                         num_views=network_config.num_views, input_size=args.input_resolution,
                         label_size=args.label_resolution, use_mask=args.use_mask, use_depth=args.use_depth)
    val_dataset = Matterport_Dataset(args.data_root, val_list,
                         transform=torchvision.transforms.Compose([
                             Stack(roll=True),
                             ToTorchFormatTensor(div=True),
                             GroupNormalize(mean_rgb, std_rgb)
                         ]),
                         num_views=network_config.num_views, input_size=args.input_resolution,
                         label_size=args.label_resolution, use_mask=args.use_mask, use_depth=args.use_depth)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=True
    )

    mapper = VPNModel(network_config)
    mapper = nn.DataParallel(mapper.cuda())

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            mapper.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(mapper.parameters(),
                          lr=args.start_lr, betas=(0.95, 0.999))

    if not os.path.isdir(args.log_root):
        os.mkdir(args.log_root)
    log_train = open(os.path.join(args.log_root, '%s.csv' % args.store_name), 'w')

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        train(train_loader, mapper, criterion, optimizer, epoch, log_train)

        if (epoch + 1) % args.ckpt_freq == 0 or epoch == args.epochs - 1:
            prec1 = eval(val_loader, mapper, criterion, log_train, epoch)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': network_config.encoder,
                'state_dict': mapper.state_dict(),
                'best_IOU': best_prec1,
            }, is_best)

def train(train_loader, mapper, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mean_iou = AverageMeter()

    mapper.train()

    end = time.time()
    for step, data in enumerate(train_loader):
        rgb_stack, target = data
        data_time.update(time.time() - end)
        target_var = target.cuda()
        input_rgb_var = torch.autograd.Variable(rgb_stack).cuda()
        output = mapper(input_rgb_var) # Shape (batch_size, label_size, label_size, num class)
        loss = criterion(output.view(-1).float(), target_var.view(-1).float())
        losses.update(loss.data[0], input_rgb_var.size(0))
        iou = calculate_iou(output, target_var)
        mean_iou.update(iou, rgb_stack.size(0))

        optimizer.zero_grad()

        loss.sum().backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            output = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch + 1, step + 1, len(train_loader))
            output += 'lr: ' + str(optimizer.param_groups[-1]['lr']) + '\t'
            output += 'Mean IOU: ' + str(mean_iou.avg.item()) + '\t'
            output += 'Mean Loss: ' + str(losses.avg.item()) + '\t'
            output += 'Data Time: '+ str(data_time.avg) + '\t' + 'Batch Time: '+str(batch_time.avg)
            print(output)
            log.write(output + '\n')
            log.flush()

def eval(val_loader, mapper, criterion, log, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    iou = AverageMeter()

    mapper.eval()

    end = time.time()
    for step, (rgb_stack, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            input_rgb_var = torch.autograd.Variable(rgb_stack).cuda()
            output = mapper(input_rgb_var)
        target_var = target.cuda()
        target_var = target_var.view(-1, 41)
        output = output.view(-1, args.num_class)
        loss = criterion(output.float(), target_var.float())
        losses.update(loss.data[0], input_rgb_var.size(0))
        iou = calculate_iou(output, target_var)
        mean_iou.update(iou, rgb_stack.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            output = 'Test: [{0}][{1}/{2}]\t'.format(epoch + 1, step + 1, len(val_loader))
            output += 'lr: ' + str(optimizer.param_groups[-1]['lr']) + '\t'
            output += 'Mean IOU: ' + str(mean_iou.avg.item()) + '\t'
            output += 'Mean Loss: ' + str(losses.avg.item()) + '\t'
            output += 'Data Time: '+ str(data_time.avg) + '\t' + 'Batch Time: '+str(batch_time.avg)
            print(output)
            log.write(output + '\n')
            log.flush()

    return mean_iou.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name), '%s/%s_best.pth.tar' % (args.root_model, args.store_name))


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.start_lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = decay

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def calculate_iou(outputs: torch.Tensor, labels: torch.Tensor):
    
    SMOOTH = 1e-4

    labels = labels.int()
    outputs = outputs.round().int()
    intersection = (outputs & labels).float().sum()
    union = (outputs | labels).float().sum()
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
        
    return iou

if __name__=='__main__':
    main()
