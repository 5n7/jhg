from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def transform_prediction(prediction, inp_dim, anchors, num_classes, use_cuda):
    """Transform prediction.

    Args:
        prediction (torch.Tensor): Prediction tensor
        inp_dim (int]): Input dimension
        anchors (list): List of anchor tuples
        num_classes (int): Number of classes
        use_cuda (bool): Whether to use CUDA

    Returns:
        torch.Tensor: Transformed tensor
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # sigmoid the centre_x, centre_y, and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if use_cuda:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if use_cuda:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # softmax the class scores
    prediction[:, :, 5 : 5 + num_classes] = torch.sigmoid((prediction[:, :, 5 : 5 + num_classes]))

    prediction[:, :, :4] *= stride

    return prediction


def get_unique_tensor(tensor):
    """Get unique tensor.

    Args:
        tensor (torch.Tensor): Target tensor

    Returns:
        torch.Tensor: Created tensor
    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2, use_cuda):
    """Returns the IoU of two bounding boxes."""
    # get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # intersection area
    if use_cuda:
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda(),) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda(),
        )
    else:
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape)
        )

    # union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def write_results(prediction, confidence, num_classes, use_cuda, nms=True, nms_conf=0.4):
    """Write results."""
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    try:
        torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except ValueError:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_a[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_a[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_a[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_a[:, :, :4]

    batch_size = prediction.size(0)

    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    for ind in range(batch_size):
        # select the image from the batch
        image_pred = prediction[ind]

        # get the class having maximum score, and the index of that class
        # get rid of num_classes softmax scores
        # add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:, 5 : 5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # get rid of the zero entries
        non_zero_ind = torch.nonzero(image_pred[:, 4])

        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

        # get the various classes detected in the image
        try:
            img_classes = get_unique_tensor(image_pred_[:, -1])
        except ValueError:
            continue

        # do NMS classwise
        for class_ in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == class_).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum
            # objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            # if nms has to be done
            if nms:
                # for each detection
                for i in range(idx):
                    # get the IoUs of all boxes that come after the one we are
                    # looking at in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1 :], use_cuda,)
                    except ValueError:
                        break

                    except IndexError:
                        break

                    # zero out all the detections that have IoU > threshold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i + 1 :] *= iou_mask

                    # remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            # concatenate the batch_id of the image to the detection
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    return output


def parse_cfg(config_path):
    """Takes a configuration file.

    Args:
        config_path (str): Path to config file.

    Returns:
        list: A list of blocks. Each blocks describes a block in the neural
        network to be built. Block is represented as a dictionary in the list.
    """
    with open(config_path) as f:
        lines = f.read().split("\n")
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != "#"]
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # this marks the start of a new block
            if block:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self):
        pass


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, use_cuda):
        x = x.data
        prediction = x
        prediction = transform_prediction(prediction, inp_dim, self.anchors, num_classes, use_cuda)
        return prediction


class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        batch_size = x.data.size(0)
        channel = x.data.size(1)
        height = x.data.size(2)
        width = x.data.size(3)

        x = (
            x.view(batch_size, channel, height, 1, width, 1)
            .expand(batch_size, channel, height, stride, width, stride)
            .contiguous()
            .view(batch_size, channel, height * stride, width * stride)
        )
        return x


def create_modules(blocks):
    # captures the information about the input and pre-processing
    net_info = blocks[0]

    module_list = nn.ModuleList()

    # indexing blocks helps with implementing route layers (skip connections)
    index = 0

    prev_filters = 3
    output_filters = []

    for x in blocks:
        module = nn.Sequential()

        if x["type"] == "net":
            continue

        # convolutional layer
        if x["type"] == "convolutional":
            # get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except (IndexError, KeyError, ValueError):
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # add the batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # check the activation
            # it is either linear or a leaky relu for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        # upsampling layer
        # use Bilinear2dUpsampling
        elif x["type"] == "upsample":
            stride = int(x["stride"])

            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # route layer
        elif x["type"] == "route":
            x["layers"] = x["layers"].split(",")

            # start of a route
            start = int(x["layers"][0])

            # end, if there exists one
            try:
                end = int(x["layers"][1])
            except (IndexError, KeyError, ValueError):
                end = 0

            # positive annotation
            if start > 0:
                start = start - index

            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)

            module.add_module("maxpool_{}".format(index), maxpool)

        # detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        else:
            assert False

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1

    return (net_info, module_list)


class Darknet(nn.Module):
    def __init__(self, config_path, use_cuda):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(config_path)
        self.use_cuda = use_cuda

        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def get_blocks(self):
        return self.blocks

    def get_module_list(self):
        return self.module_list

    def forward(self, x):
        detections = []
        modules = self.blocks[1:]
        outputs = {}  # cache the outputs for the route layer

        write = 0
        for i, module in enumerate(modules):

            module_type = module["type"]
            if module_type in ("convolutional", "upsample", "maxpool"):
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                # get the input dimensions
                inp_dim = int(self.net_info["height"])

                # get the number of classes
                num_classes = int(module["classes"])

                # output the result
                x = x.data
                x = transform_prediction(x, inp_dim, anchors, num_classes, self.use_cuda)

                if isinstance(x, int):
                    continue

                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i - 1]

        return detections

    def load_weights(self, weights_path):
        """Load weights to the.

        Args:
            weights_path str): Path to weights file.
        """
        with open(weights_path, "rb") as f:
            # the first 4 values are header information
            # 1. major version number
            # 2. minor version number
            # 3. subversion number
            # 4. images seen
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]

            # the rest of the values are the weightsg593
            # load them up
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except (IndexError, KeyError, ValueError):
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # get the number of weights of batch norm Layer
                    num_bn_biases = bn.bias.numel()

                    # load the weights
                    bn_biases = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # number of biases
                    num_biases = conv.bias.numel()

                    # load the weights
                    conv_biases = torch.from_numpy(weights[ptr : ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the
                    # model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr : ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
