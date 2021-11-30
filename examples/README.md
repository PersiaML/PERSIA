# PERSIA Examples

We provided several examples and multiple laucnher to help you quick start a *PERSIA* task.

## Honcho Launcher
[Honcho](https://github.com/nickstenning/honcho) is a tool for managing multiple processes.Current honcho launcher only support launch the PERSIA Task in single node due to some distributed environments is hard to shared across multiple nodes.

*launch example below*
```bash
cd PERSIA/examples/src/adult-income
honcho start -e .honcho.env
```

## Docker Compose Launcher

Docker [compose](https://docs.docker.com/compose/) can launch the multiple service under the swarm mode.Follow the [swarm mode](https://docs.docker.com/engine/swarm/) to adding multiple machines to swarm cluster to apply the distributed PERSIA training task.

*launcher example below*
```bash
cd PERSIA/examples/src/adult-income
make run
```