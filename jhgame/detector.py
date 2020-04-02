from pathlib import Path

import cv2
import torch
from torch.autograd import Variable

from . import config
from .darknet import Darknet, write_results
from .gamelog import get_module_logger

LOGGER = get_module_logger(__name__)


class HandDetector:
    USE_CUDA = config.DETECTION["use_cuda"]

    def __init__(
        self,
        config_path=config.DETECTION["config_path"],
        weights_path=config.DETECTION["weights_path"],
        confidence=0.25,
        nms_thresh=0.4,
        resolution=config.DETECTION["yolo_resolution"],
    ):
        """Detect hand.

        Args:
            config_path (str, optional): Path to darknet config file.
                Defaults to config.DETECTION["config_path"].
            weights_path (str, optional): Path to darknet weights file.
                Defaults to config.DETECTION["weights_path"].
            confidence (float, optional): Min confidence for detection.
                Defaults to 0.25.
            nms_thresh (float, optional): NMS threshold. Defaults to 0.4.
            resolution (int, optional): Resolution for YOLO.
                Defaults to config.DETECTION["yolo_resolution"].
        """
        self.confidence = float(confidence)
        self.nms_thresh = float(nms_thresh)
        self.inp_dim = int(resolution)

        self.num_classes = 3
        self.model = Darknet(config_path, self.USE_CUDA)

        relative_weights_path = Path(weights_path).relative_to(Path.cwd())
        LOGGER.info("Loading %s.", relative_weights_path)
        self.model.load_weights(weights_path)
        LOGGER.info("Loaded %s.", relative_weights_path)

        self.model.net_info["height"] = self.inp_dim
        if self.USE_CUDA:
            self.model.cuda()
        self.model.eval()

    @staticmethod
    def _prep_image(img, inp_dim):
        """Preprocess the image."""
        orig_im = img
        dim = orig_im.shape[1], orig_im.shape[0]
        img = cv2.resize(orig_im, (inp_dim, inp_dim))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim

    @staticmethod
    def _process_detection(hand_idx, x_min, y_min, x_max, y_max):
        """Preprocess the detection."""
        if x_min == x_max or y_min == y_max:
            return -1, x_min, y_min, x_max, y_max
        return hand_idx, x_min, y_min, x_max, y_max

    def detect(self, frame):
        """Detect hand.

        Args:
            frame (np.ndarray): BGR image

        Returns:
            tuple: Detection results
        """
        img, _, dim = self._prep_image(frame, self.inp_dim)

        if self.USE_CUDA:
            img = img.cuda()

        output = self.model(Variable(img))
        output = write_results(
            output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thresh, use_cuda=self.USE_CUDA,
        )

        hand_idx = int(output[0, 7])
        x_min = int(output[0, 1] * frame.shape[1] / config.DETECTION["yolo_resolution"])
        y_min = int(output[0, 2] * frame.shape[0] / config.DETECTION["yolo_resolution"])
        x_max = int(output[0, 3] * frame.shape[1] / config.DETECTION["yolo_resolution"])
        y_max = int(output[0, 4] * frame.shape[0] / config.DETECTION["yolo_resolution"])

        hand_idx, x_min, y_min, x_max, y_max = self._process_detection(hand_idx, x_min, y_min, x_max, y_max)

        LOGGER.debug(
            "Hand detected. (%3d,%3d) - (%3d,%3d) %d", x_min, y_min, x_max, y_max, hand_idx,
        )

        return hand_idx, x_min, y_min, x_max, y_max
