import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scr"))
from preprocessing import DetectionRateFilter

class TestDetectionRateFilter(unittest.TestCase):
 def test_test_fold_cannot_change_training_mask(self):
  train=np.array([[1.,np.nan],[2.,np.nan],[3.,4.],[4.,np.nan]])
  test=np.array([[9.,8.]])
  f=DetectionRateFilter(.75).fit(train)
  self.assertTrue(np.array_equal(f.support_mask_,[True,False]))
  self.assertEqual(f.transform(test).shape,(1,1))
 def test_unfitted_rejected(self):
  with self.assertRaises(RuntimeError): DetectionRateFilter().transform([[1.]])

if __name__ == '__main__': unittest.main()
