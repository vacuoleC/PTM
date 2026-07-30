import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scr'))
from splits import repeated_patient_splits
class T(unittest.TestCase):
 def test_count_and_overlap(self):
  ids=np.array([f'P{i}' for i in range(20)]); y=np.array([0]*10+[1]*10)
  s=list(repeated_patient_splits(ids,y,5,2,0)); self.assertEqual(len(s),10)
  self.assertTrue(all(not(set(ids[a])&set(ids[b])) for _,a,b in s))
if __name__=='__main__': unittest.main()
