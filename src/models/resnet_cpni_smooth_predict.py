'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from layers import NoisedConv2DColored as Conv2d
from layers import NoisedLinear as Linear


def conv3x3(in_planes, out_planes, stride=1, act_dim_a=None, act_dim_b=None, weight_noise=False, act_noise_a=False,
            act_noise_b=False, rank=5):
    " 3x3 convolution with padding "
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, act_dim_a=act_dim_a,
                  act_dim_b=act_dim_b, weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b,
                  rank=rank)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_dim_a=None, act_dim_b=None, weight_noise=False,
                 act_noise_a=False, act_noise_b=False, rank=5):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, act_dim_a, act_dim_b, weight_noise=weight_noise,
                             act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, act_dim_b, act_dim_b, weight_noise=weight_noise,
                             act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_dim_a=None, act_dim_b=None, weight_noise=False,
                 act_noise_a=False, act_noise_b=False, rank=5):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False, act_dim_a=act_dim_a, act_dim_b=act_dim_a,
                            weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, act_dim_a=act_dim_a,
                            act_dim_b=act_dim_b, weight_noise=weight_noise, act_noise_a=act_noise_a,
                            act_noise_b=act_noise_b, rank=rank)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False, act_dim_a=act_dim_b, act_dim_b=act_dim_b,
                            weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_dim_a=None, act_dim_b=None, weight_noise=False,
                 act_noise_a=False, act_noise_b=False, rank=5):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride, act_dim_a=act_dim_a, act_dim_b=act_dim_b,
                             weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, 1, act_dim_a=act_dim_b, act_dim_b=act_dim_b, weight_noise=weight_noise,
                             act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_dim_a=None, act_dim_b=None, weight_noise=False,
                 act_noise_a=False, act_noise_b=False, rank=5):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False, act_dim_a=act_dim_a, act_dim_b=act_dim_a,
                            weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, act_dim_a=act_dim_a,
                            act_dim_b=act_dim_b, weight_noise=weight_noise, act_noise_a=act_noise_a,
                            act_noise_b=act_noise_b, rank=rank)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False, act_dim_a=act_dim_b, act_dim_b=act_dim_b,
                            weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, width=1, num_classes=10, input_size=32, weight_noise=False, act_noise_a=False,
                 act_noise_b=False, rank=5, noise_sd=0.0, m_test=1, m_train=1, learn_noise=False):
        super(ResNet_Cifar, self).__init__()

        self.weight_noise = weight_noise
        self.act_noise_a = act_noise_a
        self.act_noise_b = act_noise_b
        self.rank = rank

        inplanes = int(16 * width)
        self.inplanes = inplanes
        self.conv1 = Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False,
                            act_dim_a=input_size, act_dim_b=input_size)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0], input_size=input_size)
        self.layer2 = self._make_layer(block, 2 * inplanes, layers[1], stride=2, input_size=input_size)
        self.layer3 = self._make_layer(block, 4 * inplanes, layers[2], stride=2, input_size=input_size // 2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = Linear(4 * inplanes * block.expansion, num_classes)
        self.num_classes = num_classes
        self.learn_noise = learn_noise
        self.noise_sd = torch.tensor(noise_sd, requires_grad=learn_noise)
        self.m_test = m_test
        self.m_train = m_train

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, input_size=32):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                       act_dim_a=input_size, act_dim_b=input_size // stride, weight_noise=self.weight_noise,
                       act_noise_a=self.act_noise_a, act_noise_b=self.act_noise_b, rank=self.rank),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, act_dim_a=input_size, act_dim_b=input_size // stride,
                  weight_noise=self.weight_noise, act_noise_a=self.act_noise_a, act_noise_b=self.act_noise_b,
                  rank=self.rank))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, act_dim_a=input_size // stride, act_dim_b=input_size // stride,
                                weight_noise=self.weight_noise, act_noise_a=self.act_noise_a,
                                act_noise_b=self.act_noise_b, rank=self.rank))

        return nn.Sequential(*layers)

    def forward(self, x, add_noise=True):
        if add_noise:
            x = x + torch.randn_like(x) * self.noise_sd

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_backward(self, data, target, criterion, factor=1.):
        output = self.forward(data)

        loss = criterion(output, target)
        loss.backward()
        return loss, output

    def expectation_forward_backward(self, data, target, criterion, factor=1.):
        if self.m_train < 2:
            output = self.forward(data)
            loss = criterion(output, target)
            loss.backward()
            return loss, output
        output_list = []
        loss_list = []
        factor = factor / self.m_train
        for _ in trange(self.m_train):
            output_i = self.forward(data)
            loss_i = factor * criterion(output_i, target)
            loss_i.backward()

            output_list.append(output_i.detach().unsqueeze(0))
            loss_list.append(loss_i.detach().unsqueeze(0))

        outs = torch.cat(output_list)
        output = outs.mean(0)
        losses = torch.cat(loss_list)
        loss = losses.sum(0)
        return loss, output

    def adv_forward_backward(self, data, target, criterion, att, eps, normalize, adv_w):
        self.eval()
        data_a, _, _ = att.generate_sample(data, target, eps, normalize=normalize)

        self.zero_grad()
        self.train()
        output = self.forward(data)
        output_a = self.forward(data_a)  # TODO: optimize

        ad_loss = criterion(output_a, target)
        reg_loss = criterion(output, target)
        loss = adv_w * ad_loss + (1 - adv_w) * reg_loss
        # tqdm.write("Reg {} Ad{} Tot {}".format(ad_loss.item(), reg_loss.item(), loss.item()))
        loss.backward()
        return loss, ad_loss, output, output_a

    def predict(self, x, output, maxk):
        _, pred = output.topk(maxk, 1, True, True)
        return self.monte_carlo_predict(x, maxk, pred)

    def monte_carlo_predict(self, x, maxk, pred):
        with torch.no_grad():
            if self.m_test == 1:
                output = self.forward(x)
                _, predictions = output.topk(maxk, 1, True, True)
                return predictions
            pred_flat = pred.view(-1)

            histogram = torch.zeros(pred_flat.shape[0], self.num_classes).to(x)

            for _ in trange(self.m_test):
                output = self.forward(x)

                _, pred_i = output.topk(maxk, 1, True, True)
                pred_i_flat = pred_i.view(-1, 1)
                histogram_values = histogram.gather(1, pred_i_flat)
                histogram = histogram.scatter(1, pred_i_flat, histogram_values + 1)
            histogram = histogram.view(pred.shape[0], pred.shape[1], self.num_classes)
            predict = torch.empty_like(pred)

            for j in range(maxk):
                predict[:, j] = torch.argmax(histogram[:, j, :], dim=1)
                histogram[np.arange(histogram.shape[0]), :, predict[:, j]] = -1

            return predict

    def to(self, *args, **kwargs):
        super(ResNet_Cifar, self).to(*args, **kwargs)
        self.noise_sd = self.noise_sd.to(*args, **kwargs).detach().requires_grad_(self.learn_noise)
        return self

class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10, weight_noise=False, act_noise_a=False, act_noise_b=False, rank=5):
        super(PreAct_ResNet_Cifar, self).__init__()

        self.weight_noise = weight_noise
        self.act_noise_a = act_noise_a
        self.act_noise_b = act_noise_b
        self.rank = rank

        self.inplanes = 16
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                       weight_noise=self.weight_noise, act_noise_a=self.act_noise_a, act_noise_b=self.act_noise_b,
                       rank=self.rank)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, weight_noise=self.weight_noise,
                  act_noise_a=self.act_noise_a, act_noise_b=self.act_noise_b, rank=self.rank))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, weight_noise=self.weight_noise, act_noise_a=self.act_noise_a,
                                act_noise_b=self.act_noise_b))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_backward(self, data, target, criterion, factor=1.):
        output = self.forward(data)

        loss = criterion(output, target)
        loss.backward()
        return loss, output

    def adv_forward_backward(self, data, target, criterion, att, eps, normalize, adv_w):
        self.eval()
        data_a, _, _ = att.generate_sample(data, target, eps, normalize=normalize)

        self.zero_grad()
        self.train()
        output = self.forward(data)
        output_a = self.forward(data_a)  # TODO: optimize

        ad_loss = criterion(output_a, target)
        reg_loss = criterion(output, target)
        loss = adv_w * ad_loss + (1 - adv_w) * reg_loss
        # tqdm.write("Reg {} Ad{} Tot {}".format(ad_loss.item(), reg_loss.item(), loss.item()))
        loss.backward()
        return loss, ad_loss, output, output_a

    def predict(self, x, output, maxk):
        _, pred = output.topk(maxk, 1, True, True)
        return pred


class ResNet_Cifar_FilteredMonteCarlo(ResNet_Cifar):
    def __init__(self, block, layers, width=1, num_classes=10, input_size=32, weight_noise=False, act_noise_a=False,
                 act_noise_b=False, rank=5, noise_sd=0.0, m_test=1, m_train=1, learn_noise=False, w_assign_fn=None):
        super(ResNet_Cifar_FilteredMonteCarlo, self).__init__(block, layers, width, num_classes, input_size, weight_noise, act_noise_a, act_noise_b, rank, noise_sd, m_test, m_train, learn_noise)
        self.w_assign_fn = w_assign_fn

    def monte_carlo_predict(self, x, maxk, pred):
        with torch.no_grad():
            if self.m_test == 1:
                output = self.forward(x)
                _, predictions = output.topk(maxk, 1, True, True)
                return predictions
            pred_flat = pred.view(-1)

            histogram = torch.zeros(pred_flat.shape[0], self.num_classes).to(x)

            outputs = [self.forward(x) for _ in trange(self.m_test)]
            if self.w_assign_fn is not None:
                weighted_outputs = self.w_assign_fn(outputs)
            else:
                weighted_outputs = [(p, 1) for p in outputs]

            for output, w in weighted_outputs:
                _, pred_i = output.topk(maxk, 1, True, True)
                if maxk == 1:
                    wk = w
                else:
                    wk = w.repeat(1,maxk).view(-1, 1)
                pred_i_flat = pred_i.view(-1, 1)
                histogram_values = histogram.gather(1, pred_i_flat)
                histogram = histogram.scatter(1, pred_i_flat, histogram_values + wk)
            histogram = histogram.view(pred.shape[0], pred.shape[1], self.num_classes)
            predict = torch.empty_like(pred)

            for j in range(maxk):
                predict[:, j] = torch.argmax(histogram[:, j, :], dim=1)
                histogram[np.arange(histogram.shape[0]), :, predict[:, j]] = -1

            return predict


class FilterByThreshold():
    def __init__(self, thresh):
        self.thresh = thresh
    
    def __call__(self, nn_outputs):
        for pred in nn_outputs:
            #2 is k (the number of top items to return)
            #1 is the dimension
            #The first true is to return the greatest results (false would mean lowest results)
            scores, _ = pred.topk(2, 1, True, True)
            scores_min_raw, _ = pred.topk(1, 1, False, True)
            score_min, score_max = scores_min_raw.view(-1, 1), scores[...,0].view(-1, 1)
            scores_scaled = (scores - score_min) / (score_max - score_min)
            diffs = scores_scaled[...,0] - scores_scaled[...,1]
            valid = torch.zeros_like(scores_min_raw)
            valid[diffs > self.thresh] = 1.
            #print('Valid predictions: {torch.sum(valid)}/{torch.numel(valid)}')
            yield pred, valid

class FilterByThresholdSoftmax():
    def __init__(self, thresh):
        self.thresh = thresh
    
    def __call__(self, nn_outputs):
        for pred in nn_outputs:
            pred_scaled = nn.functional.softmax(pred, dim=1)
            scores, _ = pred_scaled.topk(2, 1, True, True)
            diffs = scores[...,0] - scores[...,1]
            valid = torch.zeros_like(diffs)
            valid[diffs > self.thresh] = 1.
            #print('Valid predictions: {torch.sum(valid)}/{torch.numel(valid)}')
            yield pred, valid


class FilterByThresholdKPredictions():
    def __init__(self, k):
        self.k = k

    def __call__(self, nn_outputs):
            get_device = nn_outputs[0].get_device()
            m_batches = nn_outputs
            m=np.shape(m_batches)[0]
            size_items=m_batches[0].size()[0]
            
            diffs = torch.zeros((size_items, m),device=get_device)
            for i, pred in enumerate(m_batches):
                pred_scaled = nn.functional.softmax(pred, dim=1)
                scores, _ = pred_scaled.topk(2, 1, True, True)
                diffs[:,i] = scores[:,0] - scores[:,1]

            _, valid_idxs = diffs.topk(int(self.k), 1, True, True)
            #valid_list = []
            for i, _ in enumerate(m_batches):
                #valid = torch.zeros_like(pred)
                #print(valid)
                valid= torch.zeros((size_items,),device=get_device)
                #print((valid_idxs[:,:] == i).any(dim=1))
                valid[(valid_idxs[:,:] == i).any(dim=1)] = 1
                #valid_list.append(valid)             
                yield pred, valid


def resnet20_cifar(**kwargs):
    if 'vote_threshold' in kwargs:
        vote_fn = FilterByThresholdSoftmax(kwargs['vote_threshold'])
        del kwargs['vote_threshold']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [3, 3, 3], w_assign_fn=vote_fn, **kwargs)
    elif 'k_predictions' in kwargs:
        predictions_fn = FilterByThresholdKPredictions(kwargs['k_predictions'])
        del kwargs['k_predictions']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [3, 3, 3], w_assign_fn=predictions_fn, **kwargs)
    else:
        model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    if 'vote_threshold' in kwargs:
        vote_fn = FilterByThresholdSoftmax(kwargs['vote_threshold'])
        del kwargs['vote_threshold']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [5, 5, 5], w_assign_fn=vote_fn, **kwargs)
    elif 'k_predictions' in kwargs:
        predictions_fn = FilterByThresholdKPredictions(kwargs['k_predictions'])
        del kwargs['k_predictions']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [5, 5, 5], w_assign_fn=predictions_fn, **kwargs)
    else:
        model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    if 'vote_threshold' in kwargs:
        vote_fn = FilterByThresholdSoftmax(kwargs['vote_threshold'])
        del kwargs['vote_threshold']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [7, 7, 7], w_assign_fn=vote_fn, **kwargs)
    elif 'k_predictions' in kwargs:
        predictions_fn = FilterByThresholdKPredictions(kwargs['k_predictions'])
        del kwargs['k_predictions']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [7, 7, 7], w_assign_fn=predictions_fn, **kwargs)
    else:
        model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    if 'vote_threshold' in kwargs:
        vote_fn = FilterByThresholdSoftmax(kwargs['vote_threshold'])
        del kwargs['vote_threshold']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [9, 9, 9], w_assign_fn=vote_fn, **kwargs)
    elif 'k_predictions' in kwargs:
        predictions_fn = FilterByThresholdKPredictions(kwargs['k_predictions'])
        del kwargs['k_predictions']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [9, 9, 9], w_assign_fn=predictions_fn, **kwargs)
    else:
        model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    if 'vote_threshold' in kwargs:
        vote_fn = FilterByThresholdSoftmax(kwargs['vote_threshold'])
        del kwargs['vote_threshold']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [18, 18, 18], w_assign_fn=vote_fn, **kwargs)
    elif 'k_predictions' in kwargs:
        predictions_fn = FilterByThresholdKPredictions(kwargs['k_predictions'])
        del kwargs['k_predictions']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [18, 18, 18], w_assign_fn=predictions_fn, **kwargs)
    else:
        model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    if 'vote_threshold' in kwargs:
        vote_fn = FilterByThresholdSoftmax(kwargs['vote_threshold'])
        del kwargs['vote_threshold']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [200, 200, 200], w_assign_fn=vote_fn, **kwargs)
    elif 'k_predictions' in kwargs:
        predictions_fn = FilterByThresholdKPredictions(kwargs['k_predictions'])
        del kwargs['k_predictions']
        model = ResNet_Cifar_FilteredMonteCarlo(BasicBlock, [200, 200, 200], w_assign_fn=predictions_fn, **kwargs)
    else:
        model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    if 'vote_threshold' in kwargs:
        vote_fn = FilterByThresholdSoftmax(kwargs['vote_threshold'])
        del kwargs['vote_threshold']
        model = ResNet_Cifar_FilteredMonteCarlo(Bottleneck, [18, 18, 18], w_assign_fn=vote_fn, **kwargs)
    elif 'k_predictions' in kwargs:
        predictions_fn = FilterByThresholdKPredictions(kwargs['k_predictions'])
        del kwargs['k_predictions']
        model = ResNet_Cifar_FilteredMonteCarlo(Bottleneck, [18, 18, 18], w_assign_fn=predictions_fn, **kwargs)
    else:
        model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    if 'vote_threshold' in kwargs:
        vote_fn = FilterByThresholdSoftmax(kwargs['vote_threshold'])
        del kwargs['vote_threshold']
        model = ResNet_Cifar_FilteredMonteCarlo(Bottleneck, [111, 111, 111], w_assign_fn=vote_fn, **kwargs)
    elif 'k_predictions' in kwargs:
        predictions_fn = FilterByThresholdKPredictions(kwargs['k_predictions'])
        del kwargs['k_predictions']
        model = ResNet_Cifar_FilteredMonteCarlo(Bottleneck, [111, 111, 111], w_assign_fn=predictions_fn, **kwargs)
    else:
        model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    if 'vote_threshold' in kwargs:
        vote_fn = FilterByThresholdSoftmax(kwargs['vote_threshold'])
        del kwargs['vote_threshold']
        model = ResNet_Cifar_FilteredMonteCarlo(PreActBasicBlock, [18, 18, 18], w_assign_fn=vote_fn, **kwargs)
    elif 'k_predictions' in kwargs:
        predictions_fn = FilterByThresholdKPredictions(kwargs['k_predictions'])
        del kwargs['k_predictions']
        model = ResNet_Cifar_FilteredMonteCarlo(preactbasicblock, [18, 18, 18], w_assign_fn=predictions_fn, **kwargs)
    else:
        model = PreAct_ResNet_Cifar(preactbasicblock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    if 'vote_threshold' in kwargs:
        vote_fn = FilterByThresholdSoftmax(kwargs['vote_threshold'])
        del kwargs['vote_threshold']
        model = ResNet_Cifar_FilteredMonteCarlo(PreActBottleneck, [18, 18, 18], w_assign_fn=vote_fn, **kwargs)
    elif 'k_predictions' in kwargs:
        predictions_fn = FilterByThresholdKPredictions(kwargs['k_predictions'])
        del kwargs['k_predictions']
        model = ResNet_Cifar_FilteredMonteCarlo(PreActBottleneck, [18, 18, 18], w_assign_fn=predictions_fn, **kwargs)
    else:
        model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    if 'vote_threshold' in kwargs:
        vote_fn = FilterByThresholdSoftmax(kwargs['vote_threshold'])
        del kwargs['vote_threshold']
        model = ResNet_Cifar_FilteredMonteCarlo(PreActBottleneck, [111, 111, 111], w_assign_fn=vote_fn, **kwargs)
    elif 'k_predictions' in kwargs:
        predictions_fn = FilterByThresholdKPredictions(kwargs['k_predictions'])
        del kwargs['k_predictions']
        model = ResNet_Cifar_FilteredMonteCarlo(PreActBottleneck, [111, 111, 111], w_assign_fn=predictions_fn, **kwargs)
    else:
        model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model

if __name__ == '__main__':
    net = resnet20_cifar()
    y = net(torch.randn(1, 3, 32, 32))
    print(net)
    print(y.size())
