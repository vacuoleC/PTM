import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scr"))
from monitor_e2_2_remote import parse_status


class MonitorParserTests(unittest.TestCase):
    def test_parses_only_tab_separated_monitor_fields(self):
        text = "pid\t3542592\nwarning without tab\nprogress\tcompleted fold 10/50\noutput_lines\tABSENT\n"
        self.assertEqual(
            parse_status(text),
            {"pid": "3542592", "progress": "completed fold 10/50", "output_lines": "ABSENT"},
        )
