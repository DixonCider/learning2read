import learning2read
class Procedure:
    def __init__(self,proc_list,verbose=False):
        self.proc_list=proc_list.copy()
        self.verbose = verbose
        self.result_list=[{} for _ in range(len(proc_list))]
        self.summay={}
        self.var={}
        self.last_done_proc_id=-1
    def __str__(self):
        return "\n".join([
            "last_done_proc_id = %s"%(str(self.last_done_proc_id)),
            "   len(proc_list) = %d"%(len(self.proc_list)),
            "var : %s"%(str(self.var.keys())),
        ])
    def __repr__(self):
        return self.__str__()
    def append(self,proc):
        _proc = proc.copy()
        self.proc_list.append(_proc)
        return self
    def load_data(self,data_dict):
        for n,v in data_dict.items():
            self.var[n] = v # just assignment, not ensure it's a hard copy
        return self
    def input_from_var(self,vname):
        if type(vname)==str:
            return self.var[vname]
        assert type(vname)==list
        return [self.var[vn] for vn in vname]
    def output_to_var(self,vname,data):
        if type(vname)==str:
            self.var[vname]=data
            return
        assert type(vname)==list
        assert type(vname)==type(data)
        assert len(vname)==len(data)
        for i,d in enumerate(data):
            self.var[vname[i]]=d
    def run_id(self,proc_id):
        proc_dict = self.proc_list[proc_id]
        assert proc_dict['class']
        assert proc_dict['output']
        assert proc_dict['input_data']
        run_dict=proc_dict.copy()
        del run_dict['class']
        del run_dict['output']
        run_dict['input_data'] = self.input_from_var(run_dict['input_data'])
        
        result=eval('%s.run(**run_dict)'%proc_dict['class'])
        
        if type(proc_dict['output'])==list:
            assert type(result['output'])==type(proc_dict['output'])
            assert len(result['output'])==len(proc_dict['output'])
        
        self.output_to_var(proc_dict['output'], result['output'])
        del result['output']
        self.result_list[proc_id] = result
        self.last_done_proc_id = proc_id
        
    def run(self):
        st = self.last_done_proc_id+1
        nproc = len(self.proc_list)
        if st==nproc:
            print("[Procedure] nothing to run.")
            return self
        for i in range(st,nproc):
            if self.verbose:
                print("run_id(%d): %s"%(i,str(self.proc_list[i]) ))
            self.run_id(i)
        return self