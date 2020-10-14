import sys
import unittest
import torch

sys.path.append("ML/Pytorch/object_detection/metrics/")
from iou import intersection_over_union


class TestIntersectionOverUnion(unittest.TestCase):
    def setUp(self):
        # test cases we want to run
        self.t1_box1 = torch.tensor([0.8, 0.1, 0.2, 0.2])
        self.t1_box2 = torch.tensor([0.9, 0.2, 0.2, 0.2])
        self.t1_correct_iou = 1 / 7

        self.t2_box1 = torch.tensor([0.95, 0.6, 0.5, 0.2])
        self.t2_box2 = torch.tensor([0.95, 0.7, 0.3, 0.2])
        self.t2_correct_iou = 3 / 13

        self.t3_box1 = torch.tensor([0.25, 0.15, 0.3, 0.1])
        self.t3_box2 = torch.tensor([0.25, 0.35, 0.3, 0.1])
        self.t3_correct_iou = 0

        self.t4_box1 = torch.tensor([0.7, 0.95, 0.6, 0.1])
        self.t4_box2 = torch.tensor([0.5, 1.15, 0.4, 0.7])
        self.t4_correct_iou = 3 / 31

        self.t5_box1 = torch.tensor([0.5, 0.5, 0.2, 0.2])
        self.t5_box2 = torch.tensor([0.5, 0.5, 0.2, 0.2])
        self.t5_correct_iou = 1

        # (x1,y1,x2,y2) format
        self.t6_box1 = torch.tensor([2, 2, 6, 6])
        self.t6_box2 = torch.tensor([4, 4, 7, 8])
        self.t6_correct_iou = 4 / 24

        self.t7_box1 = torch.tensor([0, 0, 2, 2])
        self.t7_box2 = torch.tensor([3, 0, 5, 2])
        self.t7_correct_iou = 0

        self.t8_box1 = torch.tensor([0, 0, 2, 2])
        self.t8_box2 = torch.tensor([0, 3, 2, 5])
        self.t8_correct_iou = 0

        self.t9_box1 = torch.tensor([0, 0, 2, 2])
        self.t9_box2 = torch.tensor([2, 0, 5, 2])
        self.t9_correct_iou = 0

        self.t10_box1 = torch.tensor([0, 0, 2, 2])
        self.t10_box2 = torch.tensor([1, 1, 3, 3])
        self.t10_correct_iou = 1 / 7

        self.t11_box1 = torch.tensor([0, 0, 3, 2])
        self.t11_box2 = torch.tensor([1, 1, 3, 3])
        self.t11_correct_iou = 0.25

        self.t12_bboxes1 = torch.tensor(
            [
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 3, 2],
            ]
        )
        self.t12_bboxes2 = torch.tensor(
            [
                [3, 0, 5, 2],
                [3, 0, 5, 2],
                [0, 3, 2, 5],
                [2, 0, 5, 2],
                [1, 1, 3, 3],
                [1, 1, 3, 3],
            ]
        )
        self.t12_correct_ious = torch.tensor([0, 0, 0, 0, 1 / 7, 0.25])

        # Accept if the difference in iou is small
        self.epsilon = 0.001

    def test_both_inside_cell_shares_area(self):
        iou = intersection_over_union(self.t1_box1, self.t1_box2, box_format="midpoint")
        self.assertTrue((torch.abs(iou - self.t1_correct_iou) < self.epsilon))

    def test_partially_outside_cell_shares_area(self):
        iou = intersection_over_union(self.t2_box1, self.t2_box2, box_format="midpoint")
        self.assertTrue((torch.abs(iou - self.t2_correct_iou) < self.epsilon))

    def test_both_inside_cell_shares_no_area(self):
        iou = intersection_over_union(self.t3_box1, self.t3_box2, box_format="midpoint")
        self.assertTrue((torch.abs(iou - self.t3_correct_iou) < self.epsilon))

    def test_midpoint_outside_cell_shares_area(self):
        iou = intersection_over_union(self.t4_box1, self.t4_box2, box_format="midpoint")
        self.assertTrue((torch.abs(iou - self.t4_correct_iou) < self.epsilon))

    def test_both_inside_cell_shares_entire_area(self):
        iou = intersection_over_union(self.t5_box1, self.t5_box2, box_format="midpoint")
        self.assertTrue((torch.abs(iou - self.t5_correct_iou) < self.epsilon))

    def test_box_format_x1_y1_x2_y2(self):
        iou = intersection_over_union(self.t6_box1, self.t6_box2, box_format="corners")
        self.assertTrue((torch.abs(iou - self.t6_correct_iou) < self.epsilon))

    def test_additional_and_batch(self):
        ious = intersection_over_union(
            self.t12_bboxes1, self.t12_bboxes2, box_format="corners"
        )
        all_true = torch.all(
            torch.abs(self.t12_correct_ious - ious.squeeze(1)) < self.epsilon
        )
        self.assertTrue(all_true)


if __name__ == "__main__":
    print("Running Intersection Over Union Tests:")
    unittest.main()
