import time
import subprocess

def cmd(command):
    subp = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text = True)
    subp.wait(2)
    output, _ = subp.communicate()
    print(output)
    if subp.poll() == 0:
        print(subp.communicate()[1])
    else:
        print("失败")

cmd("nvidia-smi")
