import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scr"))
from preprocessing import DetectionRateFilter, make_preprocessing_pipeline

class TestDetectionRateFilter(unittest.TestCase):
 def test_test_fold_cannot_change_training_mask(self):
  train=np.array([[1.,np.nan],[2.,np.nan],[3.,4.],[4.,np.nan]])
  test=np.array([[9.,8.]])
  f=DetectionRateFilter(.75).fit(train)
  self.assertTrue(np.array_equal(f.support_mask_,[True,False]))
  self.assertEqual(f.transform(test).shape,(1,1))
 def test_unfitted_rejected(self):
  with self.assertRaises(RuntimeError): DetectionRateFilter().transform([[1.]])
 def test_imputer_and_scaler_are_fit_from_training_only(self):
  train=np.array([[1.,np.nan],[3.,np.nan],[5.,9.],[7.,np.nan]])
  test=np.array([[99.,100.]])
  p=make_preprocessing_pipeline(.75).fit(train)
  self.assertEqual(p.named_steps['median_imputer'].statistics_.shape[0],1)
  self.assertAlmostEqual(p.named_steps['standard_scaler'].mean_[0],4.)
  self.assertAlmostEqual(p.transform(test)[0,0],(99.-4.)/np.std([1.,3.,5.,7.]))

if __name__ == '__main__': unittest.main()
