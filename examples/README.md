# PersiaML Examples

We provided several examples and multiple laucnher to help you quick start a *PersiaML* task.

## Honcho Launcher
[Honcho](https://github.com/nickstenning/honcho) is a tool for managing multiple processes.Current honcho launcher only support launch the PersiaML Task in single node due to some distributed environments is hard to shared across multiple nodes.

*launch example below*
```bash
cd Persia/examples/honcho
pip3 install honcho
CODE_BASE=../src/getting_started/ honcho start
```

## Kubernetes Launcher
TBD

## Docker Compose Launcher

Docker [compose](https://docs.docker.com/compose/) can launch the multiple service under the swarm mode.Follow the [swarm mode](https://docs.docker.com/engine/swarm/) to adding multiple machines to swarm cluster to apply the distributed PersiaML training task.

*launcher example below*
```bash
cd Persia/examples/docker-compose
make run
```