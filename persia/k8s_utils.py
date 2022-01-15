import click

from persia.utils import resolve_binary_execute_path, run_command


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
    executable_path = resolve_binary_execute_path("gencrd")
    cmd = [
        executable_path,
        "--output",
        output,
    ]
    run_command(cmd)


@cli.command()
def operator():
    executable_path = resolve_binary_execute_path("operator")
    run_command([executable_path])


@cli.command()
@click.option(
    "--port", type=int, default="8080", help="Persia k8s schedule server port"
)
def server(
    port: int,
):
    executable_path = resolve_binary_execute_path("server")
    cmd = [
        executable_path,
        "--port",
        port,
    ]
    run_command(cmd)


if __name__ == "__main__":
    cli()
