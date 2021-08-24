import os
import click

import subprocess

REPLICA_INDEX = int(os.environ.get("JOB_ID", -1))
GPU_NUM = int(os.environ.get("GPU_NUM", -1))

@click.command()
@click.option("--node-rank", type=int, default=0, help="")
@click.option("--world-size", type=int,default=1, help="")
def launch_trainer(node_rank: int, world_size: int):
    print("launch trainer")
    
@click.command()
def launch_compose():
    print("launch compose")

@click.command()
def launch_middleware():
    print("launch middleware")

@click.command()
def launch_server():
    cmd = "echo helloworld"
    subprocess.check_call(
        cmd, shell=True
    )