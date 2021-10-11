#!/usr/bin/env bash

ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa

cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
service ssh start

cp /workspace/hadoop/core-site.xml /opt/hadoop/hadoop-3.3.1/etc/hadoop/
cp /workspace/hadoop/hdfs-site.xml /opt/hadoop/hadoop-3.3.1/etc/hadoop/
cp /workspace/hadoop/hadoop-env.sh /opt/hadoop/hadoop-3.3.1/etc/hadoop/
cp /workspace/hadoop/start-dfs.sh /opt/hadoop/hadoop-3.3.1/sbin/

mkdir -p /usr/local/hadoop/tmp
mkdir -p /usr/local/hadoop/tmp/dfs/name
mkdir -p /usr/local/hadoop/tmp/dfs/data

cd /opt/hadoop/hadoop-3.3.1 && hdfs namenode -format && sbin/start-dfs.sh