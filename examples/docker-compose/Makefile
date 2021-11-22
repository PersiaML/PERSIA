SERVICE := DEMO

process_data:
	pip3 install -r requirements.txt && python3 data_preprocess.py
	
run:
	make stop && docker stack deploy -c docker-compose.yml $(SERVICE) && make trainer

stop:
	docker stack rm $(SERVICE)

compose:
	docker service logs -f `docker stack ps $(SERVICE) | grep compose|head -n 1|awk '{print $$1}'`

trainer:
	docker service logs -f `docker stack ps $(SERVICE) | grep trainer|head -n 1|awk '{print $$1}'`

server:
	docker service logs -f `docker stack ps $(SERVICE) | grep server|head -n 1|awk '{print $$1}'`

embedding_worker:
	docker service logs -f `docker stack ps $(SERVICE) | grep embedding_worker|head -n 1|awk '{print $$1}'`

nats:
	docker service logs -f `docker stack ps $(SERVICE) | grep persia_nats_service|head -n 1|awk '{print $$1}'`