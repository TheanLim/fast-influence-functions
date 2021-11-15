import inspect 
import importlib 
import sys 
MAX_SIM_COUNT = 4 
correct_prev_simulation_count = 4
correct_prev_test_index = 1
incorrect_prev_simulation_count = 1
incorrect_prev_test_index = 15


def reload_config():
    importlib.reload(sys.modules['experiment_config'])

def get_vars(mode):
    # Reload the module first before getting info
    reload_config()
    
    if mode =="only-correct":
        return correct_prev_simulation_count, correct_prev_test_index
    else:
        return incorrect_prev_simulation_count, incorrect_prev_test_index

def set_vars(mode, 
             cur_simulation_count = None, 
             cur_test_index = None):    
    # Account for missing vars
    if (cur_simulation_count is None) or (cur_test_index is None):
        
        prev_simulation_count, prev_test_index = get_vars(mode)
        cur_simulation_count = prev_simulation_count if cur_simulation_count is None else cur_simulation_count
        cur_test_index = prev_test_index if cur_test_index is None else cur_test_index
        
    if mode =="only-correct":
        correct_prev_simulation_count = cur_simulation_count
        correct_prev_test_index = cur_test_index
        incorrect_prev_simulation_count, incorrect_prev_test_index = get_vars("only-incorrect")
    else:
        incorrect_prev_simulation_count = cur_simulation_count
        incorrect_prev_test_index = cur_test_index
        correct_prev_simulation_count, correct_prev_test_index = get_vars("only-correct")
    
    out = "import inspect \n"
    out += "import importlib \n"
    out += "import sys \n"
    out += "MAX_SIM_COUNT = 4 \n"
    
    out += "correct_prev_simulation_count = " + repr(correct_prev_simulation_count) + "\n"
    out += "correct_prev_test_index = " + repr(correct_prev_test_index) + "\n"
    
    out += "incorrect_prev_simulation_count = " + repr(incorrect_prev_simulation_count) + "\n"
    out += "incorrect_prev_test_index = " + repr(incorrect_prev_test_index) + "\n"
    
    out += "\n\n"
    out += inspect.getsource(reload_config) + "\n"    
    out += inspect.getsource(get_vars) + "\n"
    out += inspect.getsource(set_vars)
    
    f = open("experiment_config.py", "w")
    f.write(out)
    f.close()
    
    # Auto Reload the module after setting
    reload_config()
