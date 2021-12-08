SERVICE := DEMO

run:
	make stop && docker stack deploy -c docker-compose.yml $(SERVICE) && make nn_worker

stop:
	docker stack rm $(SERVICE)

data_loader:
	docker service logs -f `docker stack ps $(SERVICE) | grep data_loader|head -n 1|awk '{print $$1}'`

nn_worker:
	docker service logs -f `docker stack ps $(SERVICE) | grep nn_worker|head -n 1|awk '{print $$1}'`

server:
	docker service logs -f `docker stack ps $(SERVICE) | grep server|head -n 1|awk '{print $$1}'`

embedding_worker:
	docker service logs -f `docker stack ps $(SERVICE) | grep embedding_worker|head -n 1|awk '{print $$1}'`

nats:
	docker service logs -f `docker stack ps $(SERVICE) | grep persia_nats_service|head -n 1|awk '{print $$1}'`