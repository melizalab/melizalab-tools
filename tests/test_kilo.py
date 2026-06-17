# -*- mode: python -*-
import io

from dlab import kilo

test_oeaudio_log = """
2026-06-17 13:05:56.214688,"StartAcquisition"
2026-06-17 13:06:01.331335,"StartRecord RecDir=/home/melizalab/open-ephys/ PrependText=P397 AppendText=arc6-main"
2026-06-17 13:06:01.337147,"GetRecordingPath"
2026-06-17 13:06:01.345801,"metadata: {"animal": "P397", "experimenter": "uac6qw", "experiment": "arc6-main", "hemisphere": "L", "pen": 2, "site": 1, "x": 383, "y": -936, "z": -2619}"
2026-06-17 13:06:01.938498,"start arc607_SwC.wav"
2026-06-17 13:06:04.327607,"stop arc607_SwC.wav"
2026-06-17 13:06:05.223796,"start arc608_ScGB.wav"
2026-06-17 13:06:07.314393,"stop arc608_ScGB.wav"
2026-06-17 13:06:08.210272,"start arc600_ScGB.wav"
2026-06-17 13:06:10.599851,"stop arc600_ScGB.wav"
2026-06-17 13:06:11.495755,"start arc605_C.wav"
2026-06-17 13:06:12.989284,"stop arc605_C.wav"
2026-06-17 13:06:13.885207,"start arc606_AlCB.wav"
2026-06-17 13:06:15.975761,"stop arc606_AlCB.wav"
2026-06-17 13:06:16.871883,"start arc604_GB.wav"
2026-06-17 13:06:18.962514,"stop arc604_GB.wav"
2026-06-17 13:06:19.858412,"start arc605_ScGB.wav"
2026-06-17 13:06:21.652055,"stop arc605_ScGB.wav"
2026-06-17 13:06:22.546460,"start arc600_ScGBs.wav"
2026-06-17 13:06:23.741094,"stop arc600_ScGBs.wav"
2026-06-17 13:06:24.637218,"start arc607_GB.wav"
2026-06-17 13:06:26.727996,"stop arc607_GB.wav"
2026-06-17 13:06:27.623784,"start arc612_ScGB.wav"
2026-06-17 13:06:29.714660,"stop arc612_ScGB.wav"
2026-06-17 13:06:30.610668,"start arc608_ScCB.wav"
2026-06-17 13:06:32.701546,"stop arc608_ScCB.wav"
"""

def test_oeaudio_log_parsing():
    logfile = io.StringIO(test_oeaudio_log)
    stimuli = list(kilo.oeaudio_log_stims(logfile, 30000))
    assert len(stimuli) == 11
    assert stimuli[0].name == "arc607_SwC"
    assert stimuli[-1].name == "arc608_ScCB"
