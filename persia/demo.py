import tempfile
import os
import subprocess
import contextlib


# @contextlib.contextmanager
# def ensure_file():
#     with tempfile.NamedTemporaryFile("w") as file:
#         file.write("print('hello')")
#         file.flush()

#         yield file.name
    

# def main():
#     with ensure_file() as filename:
#         print(filename)

#         env = os.environ
#         env["PERSIA_DATALOADER_ENTRY"]= filename
#         process = subprocess.Popen(
#             ["persia-launcher", "data-loader"],
#             env=env
#         )
#         process.wait()

@contextlib.contextmanager
def catch_all():
    try:
        yield
    except Exception:
        print("catch it!")

if __name__ == "__main__":
    with catch_all():
        raise Exception("heheheh")