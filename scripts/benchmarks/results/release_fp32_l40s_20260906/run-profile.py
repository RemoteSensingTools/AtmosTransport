import subprocess,os,time,json,sys,pathlib
base=pathlib.Path(__file__).resolve().parent
julia='/home/cfranken/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia'
labels=sys.argv[1:] or ['baseline','current']
counts=tuple(map(int,os.environ.get('ATMOSTR_PROFILE_TRACERS','6,32').split(',')))
# Reject stale output before opening logs or starting any worker/monitor.
for label in labels:
    for nt in counts:
        for sample in range(6):
            target=base/(label+'-day')/('w24_tracers%d_sample%d.toml'%(nt,sample))
            if target.exists(): raise RuntimeError('Refusing stale result '+str(target))
workers={}
order=[]
suffix=os.environ.get('ATMOSTR_PROFILE_SUFFIX','')
monitor_log=open(base/('device-samples'+suffix+'.csv'),'w')
monitor=subprocess.Popen(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu','--format=csv,noheader,nounits','--loop-ms=500'],stdout=monitor_log,stderr=subprocess.STDOUT)
try:
    for label in labels:
        out=base/(label+'-day')
        out.mkdir(exist_ok=True)
        log=open(base/(label+'-profile'+suffix+'.log'),'w')
        worker_env=os.environ.copy()
        if label=='baseline': worker_env['ATMOSTR_PROFILE_LEGACY32']='1'
        worker_env['JULIA_DEPOT_PATH']=str(base/(label+'-depot'))+':/home/cfranken/.julia:'
        workers[label]=subprocess.Popen([julia,'--startup-file=no','--project='+str(base/(label+'-env')),str(base/'profile-server.jl'),str(out)],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,universal_newlines=True,env=worker_env)
    for nt in counts:
        for sample in range(6):
            sequence=labels if sample%2==0 else list(reversed(labels))
            for label in sequence:
                proc=workers[label]
                result=base/(label+'-day')/('w24_tracers%d_sample%d.toml'%(nt,sample))
                if result.exists(): raise RuntimeError('Refusing stale result '+str(result))
                print('RUNNING',label,nt,sample,flush=True)
                proc.stdin.write('%d,24,%d\n'%(nt,sample));proc.stdin.flush()
                start=time.monotonic()
                while not result.exists():
                    if proc.poll() is not None: raise RuntimeError(label+' exited '+str(proc.returncode))
                    if time.monotonic()-start>1800: raise RuntimeError(label+' timed out')
                    time.sleep(0.25)
                order.append({'label':label,'tracers':nt,'sample':sample,'elapsed_including_compile':time.monotonic()-start})
                (base/('profile-order'+suffix+'.json')).write_text(json.dumps(order,indent=2)+'\n')
                print('PASSED',label,nt,sample,flush=True)
    for proc in workers.values():
        proc.stdin.write('quit\n');proc.stdin.flush()
    for label,proc in workers.items():
        if proc.wait(timeout=120)!=0: raise RuntimeError(label+' exit failure')
    (base/('profile-order'+suffix+'.json')).write_text(json.dumps(order,indent=2)+'\n')
    print('MATCHED_PROFILE_PASSED',flush=True)
finally:
    monitor.terminate();monitor.wait(timeout=10);monitor_log.close()
    for proc in workers.values():
        if proc.poll() is None: proc.terminate()
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait()
