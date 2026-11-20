"""Execute in-process with IPython display capture; export populated HTML."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('IPYTHONDIR',str(ROOT/'.cache/ipython'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import nbformat
from nbconvert import HTMLExporter
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

def main():
    os.chdir(ROOT)
    path=ROOT/'notebooks/assignment2.ipynb'
    nb=nbformat.read(path,as_version=4)
    shell=InteractiveShell.instance()
    shell.enable_gui = lambda gui: None
    shell.run_line_magic('matplotlib','inline')
    count=0
    try:
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            count+=1
            print(f'Executing code cell {count}',flush=True)
            with capture_output() as captured:
                result=shell.run_cell(cell.source,store_history=True)
            cell.execution_count=count;cell.outputs=[]
            if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
            for out in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
            if result.error_before_exec or result.error_in_exec:
                print(captured.stdout, captured.stderr,flush=True)
                raise RuntimeError(f'Cell {count} failed') from (result.error_before_exec or result.error_in_exec)
    finally:nbformat.write(nb,path)
    html,_=HTMLExporter().from_notebook_node(nb)
    if not html.startswith('<!DOCTYPE html>'):html='<!DOCTYPE html>\n'+html
    (ROOT/'workbook.html').write_text(html)
    print('Saved executed notebook and workbook.html',flush=True)
if __name__=='__main__':main()
