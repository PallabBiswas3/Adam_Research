"""Bounded pinned selector inspection and primary-paper retrieval attempts."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/"results"/"reference-details-20261012"
path.mkdir(parents=True, exist_ok=False)
(path/"raw").mkdir()
urls = {
 "ct_selector.py": "https://raw.githubusercontent.com/Aliinton/ConfidenceTracking/85fa27da92d3171ebb3d919c51166f4396714ce2/model/SampleSelector.py",
 "ct_requirements.txt": "https://raw.githubusercontent.com/Aliinton/ConfidenceTracking/85fa27da92d3171ebb3d919c51166f4396714ce2/requirements.txt",
 "dss_selector.py": "https://raw.githubusercontent.com/Aliinton/DSS/ebbe5dd705584814d1214fe6cc1f1fb40658cf58/model/SampleSelector.py",
 "dss_requirements.txt": "https://raw.githubusercontent.com/Aliinton/DSS/ebbe5dd705584814d1214fe6cc1f1fb40658cf58/requirements.txt",
 "dss.pdf": "https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.pdf",
 "dss_supplement.pdf": "https://openaccess.thecvf.com/content/CVPR2026/supplemental/Pan_Debiased_Sample_Selection_CVPR_2026_supplemental.pdf"
}
records=[]
for filename,url in urls.items():
    record={'filename':filename,'url':url,'date':'2026-10-12','reuse':'inspection only'}
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(request,timeout=25) as response:
            data=response.read()
        (path/'raw'/filename).write_bytes(data)
        record.update(status='retrieved',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    except Exception as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
    records.append(record)
    (path/'provenance.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(filename,record['status'],flush=True)
