# zero dependency self-test script
import os,sys,subprocess
import importlib
verbose=False

def yellow(text):
    print(f"\033[33m{text}\033[0m")
def green(text):
    print(f"\033[32m{text}\033[0m")
def red(text):
    print(f"\033[31m{text}\033[0m")

def check_package_installed(package_name, err_message=None):
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    is_exists=package_name in result.stdout    
    if not is_exists:
        red(err_message)
        exit(1)
    if verbose:
        green(f"{package_name} is installed")

def check_package_directory(package_dir, err_message=None):
    import site
    sites = site.getsitepackages()
    if not sites:
        red("site.getsitepackages() returned empty list.")
        exit(1)
    if not os.path.exists(sites[0]):
        red(f"site-packages directory {sites[0]} does not exist.")
        exit(1)
    if sites[0] not in sys.path:
        red(f"site-packages directory {sites[0]} not in sys.path.")
        exit(1)
        
    root_dir = sites[0]
    site_package_dir = os.path.join(root_dir, package_dir)
    if not os.path.exists(site_package_dir):
        red(err_message)
        exit(1)
    if verbose:
        green(f"site-packages directory {site_package_dir} exists")

def self_test():

    # Check if packages installed
    check_package_installed("gai-rag-svr", "gai-rag-svr is not installed.")
    check_package_installed("gai-sdk", "gai-sdk is not installed. gai-sdk may be missing in 'tool.poetry.dependencies'.")
    
    # Check package directories
    check_package_directory("gai","site-packages/gai does not exists.")
    check_package_directory("gai/rag","site-packages/gai/rag does not exists.")
    check_package_directory("gai/rag/server","site-packages/gai/rag/server does not exists.")
    check_package_directory("gai/rag/server/api","site-packages/gai/rag/server/api does not exists. Check if gai-rag-svr/setup.py has packaged the files correctly.")
        
    green("self_test: OK")
    exit(0)

if __name__ == "__main__":
    verbose = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "--verbose":
            verbose=True
    self_test()