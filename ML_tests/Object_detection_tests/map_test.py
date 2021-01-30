import sys
import unittest
import torch

sys.path.append("ML/Pytorch/object_detection/metrics/")
from mean_avg_precision import mean_average_precision

class TestMeanAveragePrecision(unittest.TestCase):
    def setUp(self):
        # test cases we want to run
        self.t1_preds = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t1_targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t1_correct_mAP = 1

        self.t2_preds = [
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t2_targets = [
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t2_correct_mAP = 1

        self.t3_preds = [
            [0, 1, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 1, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 1, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t3_targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t3_correct_mAP = 0

        self.t4_preds = [
            [0, 0, 0.9, 0.15, 0.25, 0.1, 0.1],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]

        self.t4_targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
        self.t4_correct_mAP = 5 / 18

        self.epsilon = 1e-4

    def test_all_correct_one_class(self):
        mean_avg_prec = mean_average_precision(
            self.t1_preds,
            self.t1_targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(self.t1_correct_mAP - mean_avg_prec) < self.epsilon)

    def test_all_correct_batch(self):
        mean_avg_prec = mean_average_precision(
            self.t2_preds,
            self.t2_targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(self.t2_correct_mAP - mean_avg_prec) < self.epsilon)

    def test_all_wrong_class(self):
        mean_avg_prec = mean_average_precision(
            self.t3_preds,
            self.t3_targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=2,
        )
        self.assertTrue(abs(self.t3_correct_mAP - mean_avg_prec) < self.epsilon)

    def test_one_inaccurate_box(self):
        mean_avg_prec = mean_average_precision(
            self.t4_preds,
            self.t4_targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=1,
        )
        self.assertTrue(abs(self.t4_correct_mAP - mean_avg_prec) < self.epsilon)

    def test_all_wrong_class(self):
        mean_avg_prec = mean_average_precision(
            self.t3_preds,
            self.t3_targets,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=2,
        )
        self.assertTrue(abs(self.t3_correct_mAP - mean_avg_prec) < self.epsilon)


if __name__ == "__main__":
    print("Running Mean Average Precisions Tests:")
    unittest.main()
