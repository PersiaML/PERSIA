import os
import click
import subprocess


@click.group()
def cli():
    ...


@cli.command()
@click.option(
    "--output",
    type=str,
    default="./jobs.persia.com.yaml",
    help="Persia k8s custom resource definition description file",
)
def gencrd(
    output: str,
):
    executable_path = os.path.realpath(os.path.join(__file__, "../", "gencrd"))
    cmd = [
        executable_path,
        "--output",
        output,
    ]
    cmd = list(map(str, cmd))
    subprocess.check_call(cmd)


@cli.command()
def operator():
    executable_path = os.path.realpath(os.path.join(__file__, "../", "operator"))
    subprocess.check_call(executable_path)


@cli.command()
@click.option(
    "--port", type=int, default="8080", help="Persia k8s schedule server port"
)
def server(
    port: int,
):
    executable_path = os.path.realpath(os.path.join(__file__, "../", "server"))
    cmd = [
        executable_path,
        "--port",
        port,
    ]
    cmd = list(map(str, cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    cli()
