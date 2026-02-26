import sys, subprocess, platform

def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f"[erro ao executar {cmd}]: {e}"

def main():
    print("=== Ambiente ===")
    print("Python:", sys.version.replace("\n"," "))
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())
    print("\n=== Pip ===")
    print(run([sys.executable, "-m", "pip", "--version"]))
    print("\n=== scikit-learn (import name: sklearn) ===")
    try:
        import sklearn
        print("sklearn.__version__:", sklearn.__version__)
        print("sklearn.__file__:", sklearn.__file__)
    except Exception as e:
        print("Falha ao importar sklearn:", e)
        print("\nTente: python -m pip install scikit-learn")
    print("\nOK.")

if __name__ == "__main__":
    main()
