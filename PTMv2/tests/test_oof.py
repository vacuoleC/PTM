import sys,unittest; from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scr'))
from oof import initialise_oof,record_fold_scores
class T(unittest.TestCase):
 def test_predefined_test_only(self):
  a=pd.DataFrame({'fold':[0,0,1],'role':['test','test','test'],'patient_id':['a','b','c'],'target':[0,1,0]}); o=initialise_oof(a)
  o=record_fold_scores(o,0,['a','b'],[.1,.9]); self.assertTrue(o.score.notna().sum()==2)
  with self.assertRaises(ValueError): record_fold_scores(o,0,['a','b'],[.1,.9])
