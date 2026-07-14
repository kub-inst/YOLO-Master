import subprocess,sys
from pathlib import Path
MARKER='DDP_E2E_UNIQUE_MARKER_7F3A'
def test_torchrun_rank1_marker_reaches_parent_output(tmp_path):
 worker=tmp_path/'worker.py';logs=tmp_path/'logs'
 worker.write_text("import os\nfrom torch.distributed.elastic.multiprocessing.errors import record\n@record\ndef main():\n if int(os.environ['LOCAL_RANK'])==1: raise RuntimeError('"+MARKER+"')\nif __name__=='__main__': main()\n")
 cmd=[sys.executable,'-m','torch.distributed.run','--master_addr=127.0.0.1','--master_port=29683','--nproc_per_node=2','--log-dir',str(logs),'--tee','3',str(worker)]
 result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45)
 assert result.returncode!=0
 assert MARKER in result.stdout
 assert 'Root Cause' in result.stdout
 assert list(logs.rglob('error.json'))
