import toml,os,subprocess
from gai.scripts._scripts_utils import _get_version

base_name="gai-rag"

def _docker_run_rag(component,version="latest"):
    if component != base_name:
        print("Wrong component found.")
        return
    cmd=f"""docker run -d \
        -e DEFAULT_GENERATOR="rag-instructor-sentencepiece" \
        -e SWAGGER_URL="/doc" \
        --gpus all \
        -v ~/.gai:/app/.gai \
        -p 12036:12036 \
        --name {component} \
        --network gai-sandbox \
        kakkoii1337/{component}:{version}"""
    os.system(f"docker stop {component} && docker rm -f {component}")
    os.system(cmd)

def main():
    here=os.path.dirname(__file__)
    pyproject_dir=os.path.join(here,'..')
    dockerfile_dir=pyproject_dir
    pyproject_path=os.path.join(pyproject_dir,'pyproject.toml')

    # Get the version from the pyproject.toml file
    version = _get_version(pyproject_path=pyproject_path)
    print(f"Running {base_name}:{version}")

    # Exec the docker image
    _docker_run_rag(component=base_name,version=version)

if __name__ == "__main__":
    main()