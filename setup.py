import os
src = "/Users/qtwu/anaconda/lib/python3.6/site-packages/learning2read"
dst = "%s/learning2read/"%os.getcwd()
print("src: "+src)
print("dst: "+dst)
# os.symlink(src, dst)

def run(cmd):
    # cmd = "ln -s %s %s"%(dst,src)
    print("cmd: "+cmd)
    result=os.system(cmd)
    print("result=%d"%result)
    return result
run("rm %s"%src)
run("ln -s %s %s"%(dst,src))