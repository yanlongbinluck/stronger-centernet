from .single_stage import SingleStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class Stronger_Centernet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Stronger_Centernet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)
